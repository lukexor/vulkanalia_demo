use anyhow::{anyhow, Result};
use std::{collections::HashSet, ffi::CStr, os::raw::c_void};
use thiserror::Error;
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    prelude::v1_0::*,
    vk::{ExtDebugUtilsExtension, InstanceCreateFlags, KhrSurfaceExtension, KhrSwapchainExtension},
    window as vk_window,
};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions); // Optional debugging
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const REQUIRED_FLAGS: InstanceCreateFlags = InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
const REQUIRED_EXTENSIONS: &[&vk::ExtensionName] = &[
    &vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name,
    &vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name,
];
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_PORTABILITY_SUBSET_EXTENSION.name,
];

/// The maximum number of frames that can be processed concurrently.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkanalia Tutorial")
        .with_inner_size(LogicalSize::new(1280, 1024))
        .build(&event_loop)?;

    // App
    let mut app = unsafe { App::create(&window)? };
    let mut destroying = false;
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        log::trace!("Event: {:?}", event);
        match event {
            Event::MainEventsCleared if !destroying => {
                if let Err(err) = unsafe { app.render(&window) } {
                    log::error!("{}", err);
                    destroying = true;
                    control_flow.set_exit_with_code(1);
                }
            }
            Event::LoopDestroyed
            | Event::WindowEvent {
                event: WindowEvent::CloseRequested | WindowEvent::Destroyed,
                ..
            } => {
                if !destroying {
                    unsafe { app.destroy() }
                }
                destroying = true;
                control_flow.set_exit();
            }
            _ => (),
        }
    });
}

#[derive(Debug, Clone)]
#[must_use]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
}

#[derive(Default, Debug, Clone)]
#[must_use]
struct AppData {
    messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    // Sync Objects
    image_available_semaphor: Vec<vk::Semaphore>,
    render_finished_semaphor: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

impl App {
    unsafe fn create(window: &Window) -> Result<Self> {
        unsafe fn create_instance(
            window: &Window,
            entry: &Entry,
            data: &mut AppData,
            layers: &[*const i8],
        ) -> Result<Instance> {
            // Application
            let application_info = vk::ApplicationInfo::builder()
                .application_name(b"Vulkanalia Tutorial\0")
                .application_version(vk::make_version(1, 0, 0))
                .engine_name(b"No Engine\0")
                .engine_version(vk::make_version(1, 0, 0))
                .api_version(vk::make_version(1, 0, 0));

            // Layers
            let available_layers = entry
                .enumerate_instance_layer_properties()?
                .iter()
                .map(|l| l.layer_name)
                .collect::<HashSet<_>>();
            if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
                return Err(anyhow!("Validation layer requested but not supported."));
            }

            // Extensions
            let mut extensions = vk_window::get_required_instance_extensions(window)
                .iter()
                .chain(REQUIRED_EXTENSIONS)
                .map(|e| e.as_ptr())
                .collect::<Vec<_>>();
            if VALIDATION_ENABLED {
                extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
            }

            // Creation
            let mut instance_info = vk::InstanceCreateInfo::builder()
                .application_info(&application_info) // Optional, but can enable optimizations
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions)
                .flags(REQUIRED_FLAGS);
            let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .user_callback(Some(debug_callback));
            if VALIDATION_ENABLED {
                instance_info = instance_info.push_next(&mut debug_info);
            }

            let instance = entry.create_instance(&instance_info, None)?;
            if VALIDATION_ENABLED {
                data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
            }

            Ok(instance)
        }

        unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
            unsafe fn check_physical_device(
                instance: &Instance,
                data: &AppData,
                physical_device: vk::PhysicalDevice,
            ) -> Result<()> {
                unsafe fn check_physical_device_extensions(
                    instance: &Instance,
                    physical_device: vk::PhysicalDevice,
                ) -> Result<()> {
                    let extensions = instance
                        .enumerate_device_extension_properties(physical_device, None)?
                        .iter()
                        .map(|e| e.extension_name)
                        .collect::<HashSet<_>>();
                    DEVICE_EXTENSIONS
                        .iter()
                        .all(|e| extensions.contains(e))
                        .then_some(())
                        .ok_or_else(|| {
                            anyhow!(SuitabilityError("Missing required device extensions."))
                        })
                }

                QueueFamilyIndices::get(&instance, &data, physical_device)?;
                check_physical_device_extensions(instance, physical_device)?;

                let support = SwapchainSupport::get(instance, data, physical_device)?;
                (!support.formats.is_empty() && !support.present_modes.is_empty())
                    .then_some(())
                    .ok_or_else(|| anyhow!(SuitabilityError("Insufficient swapchain support.")))
            }

            for physical_device in instance.enumerate_physical_devices()? {
                let properties = instance.get_physical_device_properties(physical_device);
                match check_physical_device(instance, data, physical_device) {
                    Ok(_) => {
                        log::info!("Selected physical device (`{}`).", properties.device_name);
                        data.physical_device = physical_device;
                        return Ok(());
                    }
                    Err(err) => log::warn!(
                        "Skipping physical device (`{}`): {}",
                        properties.device_name,
                        err
                    ),
                }
            }
            Err(anyhow!("failed to find suitable physical device"))
        }

        unsafe fn create_logical_device(
            instance: &Instance,
            data: &mut AppData,
            layers: &[*const i8],
            indices: &QueueFamilyIndices,
        ) -> Result<Device> {
            let mut unique_indices = HashSet::new();
            unique_indices.insert(indices.graphics);
            unique_indices.insert(indices.present);

            // Queue Infos
            let queue_priorities = &[1.0];
            let queue_infos = unique_indices
                .iter()
                .map(|&i| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(i)
                        .queue_priorities(queue_priorities)
                })
                .collect::<Vec<_>>();

            // Extensions
            let extensions = DEVICE_EXTENSIONS
                .iter()
                .map(|n| n.as_ptr())
                .collect::<Vec<_>>();

            // Features
            let features = vk::PhysicalDeviceFeatures::builder();

            // Create
            let device_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions)
                .enabled_features(&features);

            let device = instance.create_device(data.physical_device, &device_info, None)?;

            // Queues
            data.graphics_queue = device.get_device_queue(indices.graphics, 0);
            data.present_queue = device.get_device_queue(indices.present, 0);

            Ok(device)
        }

        unsafe fn create_swapchain(
            window: &Window,
            instance: &Instance,
            device: &Device,
            data: &mut AppData,
            indices: &QueueFamilyIndices,
        ) -> Result<()> {
            fn get_swapchain_surface_format(
                formats: &[vk::SurfaceFormatKHR],
            ) -> vk::SurfaceFormatKHR {
                formats
                    .iter()
                    .find(|f| {
                        f.format == vk::Format::B8G8R8_SRGB
                            && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .cloned()
                    .unwrap_or_else(|| formats[0])
            }

            fn get_swapchain_present_mode(
                present_modes: &[vk::PresentModeKHR],
            ) -> vk::PresentModeKHR {
                present_modes
                    .iter()
                    .find(|&&m| m == vk::PresentModeKHR::MAILBOX)
                    .cloned()
                    .unwrap_or(vk::PresentModeKHR::FIFO)
            }

            fn get_swapchain_extent(
                window: &Window,
                capabilities: vk::SurfaceCapabilitiesKHR,
            ) -> vk::Extent2D {
                // width equal to u32::max_value means that the swapchain image resolution can
                // differ from the window resolution
                if capabilities.current_extent.width != u32::max_value() {
                    capabilities.current_extent
                } else {
                    let size = window.inner_size();
                    let min = capabilities.min_image_extent;
                    let max = capabilities.max_image_extent;
                    vk::Extent2D::builder()
                        .width(size.width.clamp(min.width, max.height))
                        .height(size.height.clamp(min.height, max.height))
                        .build()
                }
            }

            // Image
            let support = SwapchainSupport::get(instance, data, data.physical_device)?;

            let surface_format = get_swapchain_surface_format(&support.formats);
            let present_mode = get_swapchain_present_mode(&support.present_modes);
            let image_extent = get_swapchain_extent(window, support.capabilities);

            data.swapchain_format = surface_format.format;
            data.swapchain_extent = image_extent;

            let mut min_image_count = support.capabilities.min_image_count + 1;
            let max_image_count = support.capabilities.max_image_count;
            // max_image_count of 0 means no maximum
            if max_image_count != 0 && min_image_count > max_image_count {
                min_image_count = max_image_count;
            }

            let (image_sharing_mode, queue_family_indices) = if indices.graphics != indices.present
            {
                // EXCLUSIVE requires ownership passing between queue families, so we'll use
                // CONCURRENT for ease of implementation for now if the queue indices are different.
                (
                    vk::SharingMode::CONCURRENT,
                    vec![indices.graphics, indices.present],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };

            // Create
            let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(data.surface)
                .min_image_count(min_image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(image_extent)
                // Always 1 unless using stereoscopic 3D
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(image_sharing_mode)
                .queue_family_indices(&queue_family_indices)
                .pre_transform(support.capabilities.current_transform)
                // Whether to blend with other windows
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                // This defaults to null, but we're explicitly saying we don't have an old
                // swapchain and only ever create 1
                .old_swapchain(vk::SwapchainKHR::null());

            data.swapchain = device.create_swapchain_khr(&swapchain_info, None)?;
            data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;

            Ok(())
        }

        unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
            data.swapchain_image_views = data
                .swapchain_images
                .iter()
                .map(|&i| {
                    let image_view_info = vk::ImageViewCreateInfo::builder()
                        .image(i)
                        .view_type(vk::ImageViewType::_2D)
                        .format(data.swapchain_format)
                        .components(
                            vk::ComponentMapping::builder()
                                .r(vk::ComponentSwizzle::IDENTITY)
                                .g(vk::ComponentSwizzle::IDENTITY)
                                .b(vk::ComponentSwizzle::IDENTITY)
                                .a(vk::ComponentSwizzle::IDENTITY),
                        )
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        );
                    device.create_image_view(&image_view_info, None)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(())
        }

        unsafe fn create_render_pass(
            instance: &Instance,
            device: &Device,
            data: &mut AppData,
        ) -> Result<()> {
            // Attachments
            let attachments = &[vk::AttachmentDescription::builder()
                .format(data.swapchain_format)
                .samples(vk::SampleCountFlags::_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
            let color_attachments = &[vk::AttachmentReference::builder()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

            // Subpasses
            let subpasses = &[vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(color_attachments)];

            // Dependencies
            let dependencies = &[vk::SubpassDependency::builder()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

            // Create
            let render_pass_info = vk::RenderPassCreateInfo::builder()
                .attachments(attachments)
                .subpasses(subpasses)
                .dependencies(dependencies);
            data.render_pass = device.create_render_pass(&render_pass_info, None)?;
            Ok(())
        }

        unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
            unsafe fn create_shader_module(
                device: &Device,
                bytecode: &[u8],
            ) -> Result<vk::ShaderModule> {
                let bytecode = bytecode.to_vec();
                let (prefix, code, suffix) = bytecode.align_to::<u32>();
                if !prefix.is_empty() || !suffix.is_empty() {
                    return Err(anyhow!("shader bytecode is not properly aligned."));
                }
                let shader_module_info = vk::ShaderModuleCreateInfo::builder()
                    .code_size(bytecode.len())
                    .code(code);
                Ok(device.create_shader_module(&shader_module_info, None)?)
            }

            // Stages
            let vert = include_bytes!("../shaders/vert.spv");
            let frag = include_bytes!("../shaders/frag.spv");

            let vert_shader_module = create_shader_module(device, &vert[..])?;
            let frag_shader_module = create_shader_module(device, &frag[..])?;

            let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(b"main\0");
            let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(b"main\0");

            // Vertex Input State
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

            // Input Assembly State
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            // Viewport State
            let viewports = &[vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(data.swapchain_extent.width as f32)
                .height(data.swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)];
            let scissors = &[vk::Rect2D::builder()
                .offset(vk::Offset2D::default())
                .extent(data.swapchain_extent)];
            let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(viewports)
                .scissors(scissors);

            // Rasterization State
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);

            // Multisample State
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::_1);

            // Color Blend State
            let attachments = &[vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)
                .src_color_blend_factor(vk::BlendFactor::ONE) // Optional
                .dst_color_blend_factor(vk::BlendFactor::ZERO) // Optional
                .color_blend_op(vk::BlendOp::ADD) // Optional
                .src_alpha_blend_factor(vk::BlendFactor::ONE) // Optional
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO) // Optional
                .alpha_blend_op(vk::BlendOp::ADD)];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(attachments)
                .blend_constants([0.0; 4]);

            // Layout
            let layout_info = vk::PipelineLayoutCreateInfo::builder();
            data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

            let stages = &[vert_stage, frag_stage];
            let graphics_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .layout(data.pipeline_layout)
                .render_pass(data.render_pass)
                .subpass(0);

            data.pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[graphics_info], None)?
                .0;

            // Cleanup
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);

            Ok(())
        }

        unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
            data.framebuffers = data
                .swapchain_image_views
                .iter()
                .map(|&i| {
                    let attachments = &[i];
                    let framebuffer_info = vk::FramebufferCreateInfo::builder()
                        .render_pass(data.render_pass)
                        .attachments(attachments)
                        .width(data.swapchain_extent.width)
                        .height(data.swapchain_extent.height)
                        .layers(1);
                    device.create_framebuffer(&framebuffer_info, None)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(())
        }

        unsafe fn create_command_pool(
            instance: &Instance,
            device: &Device,
            data: &mut AppData,
            indices: &QueueFamilyIndices,
        ) -> Result<()> {
            let command_pool_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::empty()) // Optional
                .queue_family_index(indices.graphics);
            data.command_pool = device.create_command_pool(&command_pool_info, None)?;
            Ok(())
        }

        unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
            // Allocate
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(data.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(data.framebuffers.len() as u32);
            data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

            // Commands
            for (i, buffer) in data.command_buffers.iter().enumerate() {
                let command_buffer_info = vk::CommandBufferBeginInfo::builder();
                device.begin_command_buffer(*buffer, &command_buffer_info)?;

                let clear_values = &[vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];
                let render_pass_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(data.render_pass)
                    .framebuffer(data.framebuffers[i])
                    .render_area(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D::default())
                            .extent(data.swapchain_extent),
                    )
                    .clear_values(clear_values);

                device.cmd_begin_render_pass(
                    *buffer,
                    &render_pass_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(*buffer, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
                device.cmd_draw(*buffer, 3, 1, 0, 0);
                device.cmd_end_render_pass(*buffer);

                device.end_command_buffer(*buffer)?;
            }
            Ok(())
        }

        unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
            let semaphor_info = vk::SemaphoreCreateInfo::builder();
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                data.image_available_semaphor
                    .push(device.create_semaphore(&semaphor_info, None)?);
                data.render_finished_semaphor
                    .push(device.create_semaphore(&semaphor_info, None)?);
                data.in_flight_fences
                    .push(device.create_fence(&fence_info, None)?);
            }

            data.images_in_flight = data
                .swapchain_images
                .iter()
                .map(|_| vk::Fence::null())
                .collect();
            Ok(())
        }

        log::debug!("creating app");
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();

        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let instance = create_instance(window, &entry, &mut data, &layers)?;
        data.surface = vk_window::create_surface(&instance, window)?;
        pick_physical_device(&instance, &mut data)?;
        let indices = QueueFamilyIndices::get(&instance, &data, data.physical_device)?;
        let device = create_logical_device(&instance, &mut data, &layers, &indices)?;

        create_swapchain(window, &instance, &device, &mut data, &indices)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data, &indices)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        return Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
        });
    }

    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::max_value())?;

        let image_index = self
            .device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::max_value(),
                self.data.image_available_semaphor[self.frame],
                vk::Fence::null(),
            )?
            .0 as usize;

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::max_value())?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        let wait_semaphores = &[self.data.image_available_semaphor[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = &[self.data.render_finished_semaphor[self.frame]];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;
        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.device
            .queue_present_khr(self.data.present_queue, &present_info)?;
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn destroy(&mut self) {
        log::debug!("destroying app");

        self.device.device_wait_idle().unwrap();

        self.data
            .in_flight_fences
            .iter()
            .for_each(|&f| self.device.destroy_fence(f, None));
        self.data
            .render_finished_semaphor
            .iter()
            .for_each(|&f| self.device.destroy_semaphore(f, None));
        self.data
            .image_available_semaphor
            .iter()
            .for_each(|&f| self.device.destroy_semaphore(f, None));
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        self.data
            .framebuffers
            .iter()
            .for_each(|&f| self.device.destroy_framebuffer(f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data
            .swapchain_image_views
            .iter()
            .for_each(|&v| self.device.destroy_image_view(v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None)
        }
        self.instance.destroy_instance(None);
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
#[must_use]
pub struct SuitabilityError(pub &'static str);

#[derive(Debug, Copy, Clone)]
#[must_use]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .enumerate()
            .find(|(_, property)| property.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(i, _)| i as u32);
        let present = properties
            .iter()
            .enumerate()
            .find(|&(i, _)| {
                instance
                    .get_physical_device_surface_support_khr(
                        physical_device,
                        i as u32,
                        data.surface,
                    )
                    .is_ok()
            })
            .map(|(i, _)| i as u32);

        graphics.zip(present).map_or_else(
            || {
                Err(anyhow!(SuitabilityError(
                    "Missing required queue families."
                )))
            },
            |(graphics, present)| Ok(Self { graphics, present }),
        )
    }
}

#[derive(Debug, Clone)]
#[must_use]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).message) }.to_string_lossy();
    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::error!("({:?}) {}", msg_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("({:?}) {}", msg_type, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::debug!("({:?}) {}", msg_type, message),
        _ => log::trace!("({:?}) {}", msg_type, message),
    }
    vk::FALSE
}
