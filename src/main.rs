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

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const REQUIRED_EXTENSIONS: &[&vk::ExtensionName] = &[
    &vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name,
    &vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name,
];
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_PORTABILITY_SUBSET_EXTENSION.name,
];

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
            let mut info = vk::InstanceCreateInfo::builder()
                .application_info(&application_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions)
                .flags(InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
            let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .user_callback(Some(debug_callback));
            if VALIDATION_ENABLED {
                info = info.push_next(&mut debug_info);
            }

            let instance = entry.create_instance(&info, None)?;
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

        return Ok(Self {
            entry,
            instance,
            data,
            device,
        });
    }

    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    unsafe fn destroy(&mut self) {
        log::debug!("destroying app");
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
        let mut graphics = None;
        let mut present = None;
        for (i, properties) in properties.iter().enumerate() {
            if graphics.is_some() && present.is_some() {
                break;
            }
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                i as u32,
                data.surface,
            )? {
                present = Some(i as u32);
            }
            if properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics = Some(i as u32);
            }
        }
        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
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
