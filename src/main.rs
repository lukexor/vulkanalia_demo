use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use nalgebra_glm as glm;
use std::{
    collections::HashSet, env, ffi::CStr, fs::File, mem::size_of, os::raw::c_void,
    ptr::copy_nonoverlapping as memcpy, time::Instant,
};
use thiserror::Error;
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    prelude::v1_0::*,
    vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension},
    window as vk_window,
};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

// Optional debugging extensions
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

// Required extensions for macOS
const REQUIRED_FLAGS: vk::InstanceCreateFlags = vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
const REQUIRED_EXTENSIONS: &[&vk::ExtensionName] = &[
    &vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name,
    &vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name,
];
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_PORTABILITY_SUBSET_EXTENSION.name, // Required for macOS
];

// The maximum number of frames that can be processed concurrently.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

lazy_static! {
    static ref VERTICES: Vec<Vertex> = vec![
        Vertex::new(
            glm::vec3(-0.5, -0.5, 0.0),
            glm::vec3(1.0, 0.0, 0.0),
            glm::vec2(1.0, 0.0)
        ),
        Vertex::new(
            glm::vec3(0.5, -0.5, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
            glm::vec2(0.0, 0.0)
        ),
        Vertex::new(
            glm::vec3(0.5, 0.5, 0.0),
            glm::vec3(0.0, 0.0, 1.0),
            glm::vec2(0.0, 1.0)
        ),
        Vertex::new(
            glm::vec3(-0.5, 0.5, 0.0),
            glm::vec3(1.0, 1.0, 1.0),
            glm::vec2(1.0, 1.0)
        ),
        Vertex::new(
            glm::vec3(-0.5, -0.5, -0.5),
            glm::vec3(1.0, 0.0, 0.0),
            glm::vec2(1.0, 0.0)
        ),
        Vertex::new(
            glm::vec3(0.5, -0.5, -0.5),
            glm::vec3(0.0, 1.0, 0.0),
            glm::vec2(0.0, 0.0)
        ),
        Vertex::new(
            glm::vec3(0.5, 0.5, -0.5),
            glm::vec3(0.0, 0.0, 1.0),
            glm::vec2(0.0, 1.0)
        ),
        Vertex::new(
            glm::vec3(-0.5, 0.5, -0.5),
            glm::vec3(1.0, 1.0, 1.0),
            glm::vec2(1.0, 1.0)
        ),
    ];
}

const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];

macro_rules! time {
    ($label:ident) => {
        let mut $label = Some(Instant::now());
    };
}
macro_rules! timeLog {
    ($label:ident) => {
        match $label {
            Some(label) => log::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32()),
            None => log::warn!("Timer {} does not exist", stringify!($label)),
        };
    };
}
macro_rules! timeEnd {
    ($label:ident) => {{
        match $label.take() {
            Some(label) => log::debug!("{}: {}", stringify!($label), label.elapsed().as_secs_f32()),
            None => log::warn!("Timer {} does not exist", stringify!($label)),
        };
    }};
}

fn main() -> Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }

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
    let mut minimized = false;
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        log::trace!("Event: {:?}", event);
        match event {
            Event::MainEventsCleared if !destroying && !minimized => {
                if let Err(err) = unsafe { app.render(&window) } {
                    log::error!("{}", err);
                    destroying = true;
                    control_flow.set_exit_with_code(1);
                }
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
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

// Vulkan App
#[derive(Debug, Clone)]
#[must_use]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    start: Instant,
}

// Vulkan handles and properties
#[derive(Default, Debug, Clone)]
#[must_use]
struct AppData {
    messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    // Queues
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Buffers
    framebuffers: Vec<vk::Framebuffer>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    // Textures
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    // Depth
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    // Sync Objects
    image_available_semaphor: Vec<vk::Semaphore>,
    render_finished_semaphor: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

impl App {
    unsafe fn create(window: &Window) -> Result<Self> {
        time!(create_app);

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
                .api_version(vk::make_version(1, 3, 0));

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
                .enabled_layer_names(layers)
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

        unsafe fn pick_physical_device(
            instance: &Instance,
            data: &mut AppData,
        ) -> Result<QueueFamilyIndices> {
            unsafe fn check_physical_device(
                instance: &Instance,
                data: &AppData,
                physical_device: vk::PhysicalDevice,
            ) -> Result<QueueFamilyIndices> {
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

                let indices = QueueFamilyIndices::get(instance, data, physical_device)?;
                check_physical_device_extensions(instance, physical_device)?;

                let support = SwapchainSupport::get(instance, data, physical_device)?;
                if support.formats.is_empty() || support.present_modes.is_empty() {
                    return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
                }

                let features = instance.get_physical_device_features(physical_device);
                if features.sampler_anisotropy != vk::TRUE {
                    return Err(anyhow!(SuitabilityError("No ssampler anisotropy.")));
                }

                Ok(indices)
            }

            for physical_device in instance.enumerate_physical_devices()? {
                let properties = instance.get_physical_device_properties(physical_device);
                match check_physical_device(instance, data, physical_device) {
                    Ok(indices) => {
                        log::info!("Selected physical device (`{}`).", properties.device_name);
                        data.physical_device = physical_device;
                        return Ok(indices);
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
            // Queue Infos
            let mut unique_indices = HashSet::new();
            unique_indices.insert(indices.graphics);
            unique_indices.insert(indices.present);

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
            let features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

            // Create
            let device_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_layer_names(layers)
                .enabled_extension_names(&extensions)
                .enabled_features(&features);

            let device = instance.create_device(data.physical_device, &device_info, None)?;

            // Queues
            data.graphics_queue = device.get_device_queue(indices.graphics, 0);
            data.present_queue = device.get_device_queue(indices.present, 0);

            Ok(device)
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
        let indices = pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&instance, &mut data, &layers, &indices)?;

        Self::create_swapchain(window, &instance, &device, &mut data, &indices)?;
        Self::create_swapchain_image_views(&device, &mut data)?;
        Self::create_render_pass(&instance, &device, &mut data)?;
        Self::create_descriptor_set_layout(&device, &mut data)?;
        Self::create_pipeline(&device, &mut data)?;
        Self::create_command_pool(&instance, &device, &mut data, &indices)?;
        Self::create_texture_image(&instance, &device, &mut data)?;
        Self::create_texture_image_view(&device, &mut data)?;
        Self::create_texture_sampler(&device, &mut data)?;
        Self::create_depth_objects(&instance, &device, &mut data)?;
        Self::create_vertex_buffer(&instance, &device, &mut data)?;
        Self::create_index_buffer(&instance, &device, &mut data)?;
        Self::create_uniform_buffers(&instance, &device, &mut data)?;
        Self::create_framebuffers(&device, &mut data)?;
        Self::create_descriptor_pool(&device, &mut data)?;
        Self::create_descriptor_sets(&device, &mut data)?;
        Self::create_command_buffers(&device, &mut data)?;
        Self::create_sync_objects(&device, &mut data)?;

        timeEnd!(create_app);

        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
        })
    }

    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        time!(render_time);

        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device
            .wait_for_fences(&[in_flight_fence], true, u64::max_value())?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::max_value(),
            self.data.image_available_semaphor[self.frame],
            vk::Fence::null(),
        );
        let image_index = match result {
            Ok((index, _)) => index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(err) => return Err(anyhow!(err)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device
                .wait_for_fences(&[image_in_flight], true, u64::max_value())?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;
        self.update_uniform_buffer(image_index)?;

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
        timeEnd!(render_time);

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = matches!(
            result,
            Ok(vk::SuccessCode::SUBOPTIMAL_KHR) | Err(vk::ErrorCode::OUT_OF_DATE_KHR)
        );
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(err) = result {
            return Err(anyhow!(err));
        }
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn get_memory_type_index(
        instance: &Instance,
        data: &AppData,
        properties: vk::MemoryPropertyFlags,
        requirements: vk::MemoryRequirements,
    ) -> Result<u32> {
        let memory = instance.get_physical_device_memory_properties(data.physical_device);
        (0..memory.memory_type_count)
            .find(|&i| {
                let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                let memory_type = memory.memory_types[i as usize];
                suitable && memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| anyhow!("failed to find suitable memory type."))
    }

    unsafe fn create_buffer(
        instance: &Instance,
        device: &Device,
        data: &AppData,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        // Buffer
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&buffer_info, None)?;

        // Memory
        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(Self::get_memory_type_index(
                instance,
                data,
                properties,
                requirements,
            )?);

        // Allocate
        let buffer_memory = device.allocate_memory(&memory_info, None)?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
    }

    unsafe fn begin_single_time_cmd(device: &Device, data: &AppData) -> Result<vk::CommandBuffer> {
        // Allocate
        let command_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(data.command_pool)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&command_allocate_info)?[0];

        // Commands
        let command_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &command_begin_info)?;

        Ok(command_buffer)
    }

    unsafe fn end_single_time_cmd(
        device: &Device,
        data: &AppData,
        command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        device.end_command_buffer(command_buffer)?;

        // Submit
        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        device.queue_submit(data.graphics_queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;

        // Cleanup
        device.free_command_buffers(data.command_pool, command_buffers);

        Ok(())
    }

    unsafe fn copy_buffer(
        device: &Device,
        data: &AppData,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        let command_buffer = Self::begin_single_time_cmd(device, data)?;

        let regions = vk::BufferCopy::builder().size(size);
        device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

        Self::end_single_time_cmd(device, data, command_buffer)?;

        Ok(())
    }

    unsafe fn copy_buffer_to_image(
        device: &Device,
        data: &AppData,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let command_buffer = Self::begin_single_time_cmd(device, data)?;

        let subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource)
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        Self::end_single_time_cmd(device, data, command_buffer)?;

        Ok(())
    }

    unsafe fn transition_image_layout(
        device: &Device,
        data: &AppData,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> Result<()> {
        let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                ),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                ),
                _ => return Err(anyhow!("Unsupported image layout transition!")),
            };
        let command_buffer = Self::begin_single_time_cmd(device, data)?;

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        Self::end_single_time_cmd(device, data, command_buffer)?;

        Ok(())
    }

    unsafe fn create_vertex_buffer(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let size = (size_of::<Vertex>() * VERTICES.len()) as u64;

        // Staging Buffer
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        // Copy Staging
        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
        memcpy(VERTICES.as_ptr(), memory.cast(), VERTICES.len());
        device.unmap_memory(staging_buffer_memory);

        // Vertex Buffer
        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        data.vertex_buffer = vertex_buffer;
        data.vertex_buffer_memory = vertex_buffer_memory;

        // Copy Vertex
        Self::copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

        // Cleanup
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn create_index_buffer(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let size = (size_of::<u16>() * INDICES.len()) as u64;

        // Staging Buffer
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        // Copy Staging
        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
        memcpy(INDICES.as_ptr(), memory.cast(), INDICES.len());
        device.unmap_memory(staging_buffer_memory);

        // Index Buffer
        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        data.index_buffer = index_buffer;
        data.index_buffer_memory = index_buffer_memory;

        // Copy Index
        Self::copy_buffer(device, data, staging_buffer, index_buffer, size)?;

        // Cleanup
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let time = self.start.elapsed().as_secs_f32();
        let model = glm::rotate(
            &glm::identity(),
            time * glm::radians(&glm::vec1(90.0))[0],
            &glm::vec3(0.0, 0.0, 1.0),
        );
        let view = glm::look_at(
            &glm::vec3(2.0, 2.0, 2.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 0.0, 1.0),
        );
        let vk::Extent2D { width, height } = self.data.swapchain_extent;
        let mut proj = glm::perspective_rh_zo(
            width as f32 / height as f32,
            glm::radians(&glm::vec1(45.0))[0],
            0.1,
            10.0,
        );
        proj[(1, 1)] *= -1.0; // Y-axis is inverted in Vulkan

        let ubo = UniformBufferObject { model, view, proj };

        let uniform_buffer_memory = self.data.uniform_buffers_memory[image_index];
        let memory = self.device.map_memory(
            uniform_buffer_memory,
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);
        self.device.unmap_memory(uniform_buffer_memory);
        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;

        self.destroy_swapchain();

        let indices =
            QueueFamilyIndices::get(&self.instance, &self.data, self.data.physical_device)?;
        Self::create_swapchain(
            window,
            &self.instance,
            &self.device,
            &mut self.data,
            &indices,
        )?;
        Self::create_swapchain_image_views(&self.device, &mut self.data)?;
        Self::create_render_pass(&self.instance, &self.device, &mut self.data)?;
        Self::create_pipeline(&self.device, &mut self.data)?;
        Self::create_depth_objects(&self.instance, &self.device, &mut self.data)?;
        Self::create_framebuffers(&self.device, &mut self.data)?;
        Self::create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        Self::create_descriptor_pool(&self.device, &mut self.data)?;
        Self::create_descriptor_sets(&self.device, &mut self.data)?;
        Self::create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.device
            .destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
        self.device
            .free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data
            .uniform_buffers_memory
            .iter()
            .for_each(|&m| self.device.free_memory(m, None));
        self.data
            .uniform_buffers
            .iter()
            .for_each(|&b| self.device.destroy_buffer(b, None));
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
    }

    unsafe fn destroy(&mut self) {
        log::debug!("destroying app");

        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

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
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None)
        }

        self.instance.destroy_instance(None);
    }

    unsafe fn create_swapchain(
        window: &Window,
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
        indices: &QueueFamilyIndices,
    ) -> Result<()> {
        fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
            formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8_SRGB
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .cloned()
                .unwrap_or_else(|| formats[0])
        }

        fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
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

        let (image_sharing_mode, queue_family_indices) = if indices.graphics != indices.present {
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
            .map(|&image| {
                Self::create_image_view(
                    device,
                    image,
                    data.swapchain_format,
                    vk::ImageAspectFlags::COLOR,
                )
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
        let color_attachment = vk::AttachmentDescription::builder()
            .format(data.swapchain_format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_attachment_refs = &[vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let depth_stencil_attachment = vk::AttachmentDescription::builder()
            .format(Self::get_depth_format(instance, data)?)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        // Subpasses
        let subpasses = &[vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachment_refs)
            .depth_stencil_attachment(&depth_stencil_attachment_ref)];

        // Dependencies
        let dependencies = &[vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )];

        // Create
        let attachments = &[color_attachment, depth_stencil_attachment];
        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        data.render_pass = device.create_render_pass(&render_pass_info, None)?;

        Ok(())
    }

    unsafe fn create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo_binding, sampler_binding];
        let descriptor_set_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        data.descriptor_set_layout =
            device.create_descriptor_set_layout(&descriptor_set_info, None)?;

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
        let binding_descriptions = &[Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

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
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE) // Because Y-axis is inverted in Vulkan
            .depth_bias_enable(false);

        // Multisample State
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::_1);

        // Depth Stencil State
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0) // Optional
            .max_depth_bounds(1.0) // Optional
            .stencil_test_enable(false);

        // Color Blend State
        let attachments = &[vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0; 4]);

        // Layout
        let set_layouts = &[data.descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(set_layouts);
        data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        // Create
        let stages = &[vert_stage, frag_stage];
        let graphics_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
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
            .map(|&image_view| {
                let attachments = &[image_view, data.depth_image_view];
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

    unsafe fn create_image(
        instance: &Instance,
        device: &Device,
        data: &AppData,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(vk::SampleCountFlags::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = device.create_image(&image_info, None)?;

        let requirements = device.get_image_memory_requirements(image);
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(Self::get_memory_type_index(
                instance,
                data,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                requirements,
            )?);
        let image_memory = device.allocate_memory(&memory_info, None)?;
        device.bind_image_memory(image, image_memory, 0)?;

        Ok((image, image_memory))
    }

    unsafe fn create_texture_image(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        let image = File::open("resources/texture.png")?;

        let decoder = png::Decoder::new(image);
        let mut reader = decoder.read_info()?;
        let &png::Info { width, height, .. } = reader.info();

        let mut pixels = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut pixels)?;

        let size = pixels.len() as u64;
        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            device,
            data,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory =
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

        memcpy(pixels.as_ptr(), memory.cast(), pixels.len());
        device.unmap_memory(staging_buffer_memory);

        let (texture_image, texture_image_memory) = Self::create_image(
            instance,
            device,
            data,
            width,
            height,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        data.texture_image = texture_image;
        data.texture_image_memory = texture_image_memory;

        Self::transition_image_layout(
            device,
            data,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;

        Self::copy_buffer_to_image(
            device,
            data,
            staging_buffer,
            data.texture_image,
            width,
            height,
        )?;

        Self::transition_image_layout(
            device,
            data,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(())
    }

    unsafe fn create_image_view(
        device: &Device,
        image: vk::Image,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        let image_view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspects)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        Ok(device.create_image_view(&image_view_info, None)?)
    }

    unsafe fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
        data.texture_image_view = Self::create_image_view(
            device,
            data.texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
        )?;
        Ok(())
    }

    unsafe fn create_texture_sampler(device: &Device, data: &mut AppData) -> Result<()> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);
        data.texture_sampler = device.create_sampler(&sampler_info, None)?;
        Ok(())
    }

    unsafe fn get_supported_format(
        instance: &Instance,
        data: &AppData,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format> {
        candidates
            .iter()
            .find(|&&format| {
                let properties =
                    instance.get_physical_device_format_properties(data.physical_device, format);
                match tiling {
                    vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => {
                        properties.optimal_tiling_features.contains(features)
                    }
                    _ => false,
                }
            })
            .cloned()
            .ok_or_else(|| anyhow!("Failed to find supported format!"))
    }

    unsafe fn get_depth_format(instance: &Instance, data: &AppData) -> Result<vk::Format> {
        let candidates = &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        Self::get_supported_format(
            instance,
            data,
            candidates,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    unsafe fn create_depth_objects(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        // Depth Image
        let format = Self::get_depth_format(instance, data)?;
        let (depth_image, depth_image_memory) = Self::create_image(
            instance,
            device,
            data,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        data.depth_image = depth_image;
        data.depth_image_memory = depth_image_memory;

        // Depth Image View
        data.depth_image_view = Self::create_image_view(
            device,
            data.depth_image,
            format,
            vk::ImageAspectFlags::DEPTH,
        )?;

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

    unsafe fn create_uniform_buffers(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
    ) -> Result<()> {
        data.uniform_buffers.clear();
        data.uniform_buffers_memory.clear();

        let size = size_of::<UniformBufferObject>() as u64;
        for _ in 0..data.swapchain_images.len() {
            let (uniform_buffer, uniform_buffer_memory) = Self::create_buffer(
                instance,
                device,
                data,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
            )?;
            data.uniform_buffers.push(uniform_buffer);
            data.uniform_buffers_memory.push(uniform_buffer_memory);
        }

        Ok(())
    }

    unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
        let count = data.swapchain_images.len() as u32;

        let ubo_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(count);
        let sampler_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(count);

        let pool_sizes = &[ubo_size, sampler_size];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(count);

        data.descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

        Ok(())
    }

    unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
        //  Allocate
        let count = data.swapchain_images.len();
        let layouts = vec![data.descriptor_set_layout; count];
        let descriptor_set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(data.descriptor_pool)
            .set_layouts(&layouts);

        data.descriptor_sets = device.allocate_descriptor_sets(&descriptor_set_info)?;

        // Update
        for i in 0..count {
            let descriptor_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(data.uniform_buffers[i])
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64);

            let buffer_info = &[descriptor_buffer_info];
            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(data.descriptor_sets[i])
                .dst_binding(0) // points to shader.vert binding
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);

            let descriptor_image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(data.texture_image_view)
                .sampler(data.texture_sampler);

            let image_info = &[descriptor_image_info];
            let sampler_write = vk::WriteDescriptorSet::builder()
                .dst_set(data.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info);

            device.update_descriptor_sets(
                &[ubo_write, sampler_write],
                &[] as &[vk::CopyDescriptorSet],
            );
        }
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
            let command_begin_info = vk::CommandBufferBeginInfo::builder();
            device.begin_command_buffer(*buffer, &command_begin_info)?;

            let color_clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };
            let depth_clear_value = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            };
            let clear_values = &[color_clear_value, depth_clear_value];
            let render_pass_info = vk::RenderPassBeginInfo::builder()
                .render_pass(data.render_pass)
                .framebuffer(data.framebuffers[i])
                .render_area(
                    vk::Rect2D::builder()
                        .offset(vk::Offset2D::default())
                        .extent(data.swapchain_extent),
                )
                .clear_values(clear_values);

            device.cmd_begin_render_pass(*buffer, &render_pass_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(*buffer, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
            device.cmd_bind_vertex_buffers(*buffer, 0, &[data.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(*buffer, data.index_buffer, 0, vk::IndexType::UINT16);
            device.cmd_bind_descriptor_sets(
                *buffer,
                vk::PipelineBindPoint::GRAPHICS,
                data.pipeline_layout,
                0,
                &[data.descriptor_sets[i]],
                &[],
            );
            device.cmd_draw_indexed(*buffer, INDICES.len() as u32, 1, 0, 0, 0);
            device.cmd_end_render_pass(*buffer);

            device.end_command_buffer(*buffer)?;
        }
        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[must_use]
struct Vertex {
    pos: glm::Vec3,
    color: glm::Vec3,
    tex_coord: glm::Vec2,
}

impl Vertex {
    fn new(pos: glm::Vec3, color: glm::Vec3, tex_coord: glm::Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0) // points to shader.vert location
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1) // points to shader.vert location
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<glm::Vec2>() as u32)
            .build();
        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2) // points to shader.vert location
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<glm::Vec2>() + size_of::<glm::Vec3>()) as u32)
            .build();
        [pos, color, tex_coord]
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[must_use]
struct UniformBufferObject {
    model: glm::Mat4,
    view: glm::Mat4,
    proj: glm::Mat4,
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
