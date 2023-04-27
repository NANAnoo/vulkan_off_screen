#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/fwd.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "glm/matrix.hpp"
#include "glm/trigonometric.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <volk/volk.h>

#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <chrono>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "RenderingModel.hpp"
#include "RenderProgram.hpp"
#include "VkUBO.hpp"
#include "VkConstant.hpp"
#include "FpsController.hpp"
#include "VkColorAttachments.hpp"
#include "GaussFilter.hpp"

namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw1/shaders/*. 
#		define SHADERDIR_ "assets/cw3/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";

		constexpr char const* kScreenVertShaderPath = SHADERDIR_ "fullscreen.vert.spv";
		constexpr char const* kScreenFragShaderPath = SHADERDIR_ "screen.frag.spv";
		constexpr char const* kToneFragShaderPath = SHADERDIR_ "tone.frag.spv";
		constexpr char const* kBloomVFragShaderPath = SHADERDIR_ "bloom_v.frag.spv";
		constexpr char const* kBloomHFragShaderPath = SHADERDIR_ "bloom_h.frag.spv";
#		undef SHADERDIR_

		constexpr char const* kModelPath = "assets/cw3/ship.comp5822mesh";

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;

		const auto kDepthFormat = VK_FORMAT_D32_SFLOAT;

		enum ShadingMode{
			Basic,
			Tone,
			Bloom
		};
	}

	// Local types/structures:
	// Uniform data
	namespace glsl
	{
		struct SceneUniform {
			glm::mat4 M;
			glm::mat4 V;
			glm::mat4 P;
			glm::vec4 camPos;
		};

		struct LightInfo {
			glm::vec4 position;
			glm::vec4 color;
		};

		struct Constants {
			int filter_width;
		};
	}

	// Local functions:
	lut::RenderPass create_render_pass( lut::VulkanWindow const& );

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const&, 
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		std::vector<VkColorAttachments> const&,
		VkImageView
	);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer( 
		lut::VulkanWindow const&,
		lut::Allocator const& 
	);

	lut::QueryPool create_query_pool( 
		lut::VulkanWindow const&
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		FpsController &aController,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight
	);

	void record_commands(
		VkCommandBuffer aCmdBuff, 
		VkRenderPass aRenderPass, 
		VkFramebuffer aFramebuffer, 
		VkExtent2D const& aImageExtent,
		std::function<void(VkCommandBuffer)> const&,
		std::function<void(VkCommandBuffer)> const&
	);

	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);

	void present_results( 
		VkQueue, 
		VkSwapchainKHR, 
		std::uint32_t aImageIndex, 
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);

	// draw helpers
}

#define IS_KEY_DOWN(action) (GLFW_PRESS == action || GLFW_REPEAT == action)

int main(int argc, char **argv) try
{	
	auto level = lut::DeviceLevel::Maximum;
	if (argc > 1 && std::string(argv[1]) == "low") {
		level = lut::DeviceLevel::Minimum;
		std::cout << "Use worse device" << std::endl;
	}
	// Create Vulkan Window
	auto window = lut::make_vulkan_window(level);

	// Configure the GLFW window
	static bool enableMouseNavigation = false;
	static FpsController sController({-3, 0, -20}, {0, 180, 0}, 5, 4);
	static float cursor_x = 0, cursor_y = 0, offset_x = 0, offset_y = 0;
	static glm::vec3 sLightPosition = glm::vec3(0, 8, -1);
	static float sLightIntensity = 1.0f;
	static cfg::ShadingMode sCurrentMode = cfg::Basic;
	static bool shouldGeneratePipeLine = true;
	static bool modeChanged = true;
	glfwSetKeyCallback( window.window, 
	[]( GLFWwindow* aWindow, int aKey, int, int aAction, int){
		if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
		{	// close window
			glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
		} // move control: 
		if (aAction == GLFW_RELEASE) {
			sController.onKeyUp(aKey);
			if (aKey == GLFW_KEY_1) {
				// normal mode
				modeChanged = (sCurrentMode != cfg::Basic);
				sCurrentMode = cfg::Basic;
			} else if (aKey == GLFW_KEY_2) {
				// normal mode
				modeChanged = (sCurrentMode != cfg::Tone);
				sCurrentMode = cfg::Tone;
			} else if (aKey == GLFW_KEY_3) {
				// normal mode
				modeChanged = (sCurrentMode != cfg::Bloom);
				sCurrentMode = cfg::Bloom;
			}
			if (aKey == GLFW_KEY_UP) {
				// normal mode
				sLightPosition += glm::vec3(0, 0.1, 0);
			} else if (aKey == GLFW_KEY_DOWN) {
				sLightPosition -= glm::vec3(0, 0.1, 0);
			} else if (aKey == GLFW_KEY_LEFT) {
				sLightIntensity *= 0.5f;
			} else if (aKey == GLFW_KEY_RIGHT) {
				sLightIntensity *= 2.f;
			}
		} else if (aAction == GLFW_PRESS) {
			sController.onKeyPress(aKey);
		}
	});

	glfwSetMouseButtonCallback( window.window, 
		[]( GLFWwindow* window, int button, int action, int){
			if( GLFW_MOUSE_BUTTON_RIGHT == button && GLFW_RELEASE == action)
			{
				// change mode
				enableMouseNavigation = !enableMouseNavigation;
				glfwSetInputMode(window, GLFW_CURSOR, enableMouseNavigation ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
			}
		});

	glfwSetCursorPosCallback(window.window, 
		[](GLFWwindow *window, double xpos, double ypos){
			offset_x = float(xpos) - cursor_x;
			offset_y = float(ypos) - cursor_y;
			cursor_x = float(xpos);
			cursor_y = float(ypos);
			if (enableMouseNavigation) {
				sController.onCursorMove(offset_x, offset_y);
			}
		});

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator( window );

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass( window );

	// create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	// create query pool
	lut::QueryPool queryPool = create_query_pool(window);

	// a sampler
	lut::Sampler anisotropicSampler = lut::create_default_sampler(window);
	if (window.currentDeviceFeatures.samplerAnisotropy) {
		auto maxAnisotropy = window.maxAnisotropy;
		std::printf("Current device can support anisotropicSampler, maxSamplerAnisotropy is %f", maxAnisotropy);
		anisotropicSampler = lut::create_anisotropic_sampler(window, maxAnisotropy);
	} else {
		std::printf("Current device not support anisotropicSampler! Use defualt sampler");
	}

	lut::Sampler screen_sampler = lut::create_screen_sampler(window);
	 	
	// create all shaders
	lut::ShaderModule baseVert = lut::load_shader_module(window, cfg::kVertShaderPath);
	lut::ShaderModule baseFrag = lut::load_shader_module(window, cfg::kFragShaderPath);
	lut::ShaderModule screenVert = lut::load_shader_module(window, cfg::kScreenVertShaderPath);
	lut::ShaderModule screenFrag = lut::load_shader_module(window, cfg::kScreenFragShaderPath);
	lut::ShaderModule toneFrag = lut::load_shader_module(window, cfg::kToneFragShaderPath);
	lut::ShaderModule bloom_vFrag = lut::load_shader_module(window, cfg::kBloomVFragShaderPath);
	lut::ShaderModule bloom_hFrag = lut::load_shader_module(window, cfg::kBloomHFragShaderPath);

	// create scene ubo
	VkUBO<glsl::SceneUniform> sceneUBO(window, allocator, dpool, VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_GEOMETRY_BIT, 0);
	sceneUBO.data = std::make_unique<glsl::SceneUniform>();
	
	// create light ubo
	VkUBO<glsl::LightInfo> lightUBO(window, allocator, dpool, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	lightUBO.data = std::make_unique<glsl::LightInfo>();
	lightUBO.data->position = glm::vec4(0, 3, 0, 1);
	lightUBO.data->color = glm::vec4(1, 1, 1, 1) * sLightIntensity;

	// create gauss filter ubo
	const int kGaussFilterSize = 22; // 44 / 2, WHICH IS THE TAP SIZE
	VkSSBO<GaussFilter1D<kGaussFilterSize>> gaussSSBO(window, allocator, dpool, VK_SHADER_STAGE_FRAGMENT_BIT, 0);
	gaussSSBO.data = std::make_unique<GaussFilter1D<kGaussFilterSize>>(9.f);

	VkConstant<glsl::Constants> constants(0, VK_SHADER_STAGE_FRAGMENT_BIT);
	glsl::Constants aConstants{GaussFilter1D<kGaussFilterSize>::getArraySize() / 2};

	// pipe generator, set up basic informations
	PipeLineGenerator basicPipeGen;
	basicPipeGen
	.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height))
	.addDescLayout(sceneUBO.layout.handle)
	.addDescLayout(lightUBO.layout.handle)
	.enableBlend(false)
	.setCullMode(VK_CULL_MODE_BACK_BIT)
	.setPolyGonMode(VK_POLYGON_MODE_FILL)
	.enableDepthTest(true)
	.setRenderMode(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	// screen pipe generator
	PipeLineGenerator screenPipeGen;
	screenPipeGen
	.enableBlend(false)
	.setCullMode(VK_CULL_MODE_NONE)
	.setPolyGonMode(VK_POLYGON_MODE_FILL)
	.enableDepthTest(false)
	.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height))
	.setRenderMode(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	// cache all pipeline
	std::unordered_map<cfg::ShadingMode, std::vector<RenderPipeLine>> pipelineMap;

	// create frambuffer and depth buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer( window, allocator );

	// create off screen attachments
	std::vector<VkColorAttachments> offScreenAttachments(2);
	for (auto &attach : offScreenAttachments) {
		attach.addAttachment(VK_FORMAT_R16G16B16A16_SFLOAT);
		attach.createAttachmentLayout(window);
		attach.setFrameSize(window.swapchainExtent);
		attach.create(window, allocator, dpool, renderPass, screen_sampler.handle);
	}

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers( window, renderPass.handle, framebuffers, offScreenAttachments, depthBufferView.handle);
	
	// create descriptor set for off screen
	lut::CommandPool cpool = lut::create_command_pool( window, 
														VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | 
														VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );
	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;
	
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		cbuffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		cbfences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	// Load data
	std::shared_ptr<RenderingModel> model = std::make_shared<RenderingModel>();
	{
		auto tmp = load_baked_model(cfg::kModelPath);
		model->load(window, allocator, dpool, tmp);
		model->createTextureSetsWith(window, dpool.handle, screen_sampler.handle);
		// upload data
		RenderingModel::uploadScope(
			window, 
			[&model](VkCommandBuffer cmd){
				model->upload(cmd);
			}
		);
	}

	// Application main loop
	bool recreateSwapchain = false;
	auto previous = std::chrono::system_clock::now();
	while (!glfwWindowShouldClose(window.window)) {
		// Let GLFW process events.
		glfwPollEvents(); // or: glfwWaitEvents()
		auto now = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous).count() / 1000000000.f;
		previous = now;
		// update control info
		sController.update(duration);
		lightUBO.data->position = glm::vec4(sLightPosition, 1);
		lightUBO.data->color = glm::vec4(1, 1, 1, 1) * sLightIntensity;
		// update uniforms
		update_scene_uniforms(
			*sceneUBO.data.get(), 
			sController,
			window.swapchainExtent.width, 
			window.swapchainExtent.height
		);
		// Recreate swap chain
		if (recreateSwapchain)
		{
			//We need to destroy several objects, which may still be in use by the GPU
			vkDeviceWaitIdle( window.device );

			// Recreate them
			auto const changes = recreate_swapchain( window );

			if( changes.changedFormat ) {
				renderPass = create_render_pass( window );
			}
			if( changes.changedSize ) {
				std::tie(depthBuffer,depthBufferView) = create_depth_buffer( window, allocator );
				basicPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));
				screenPipeGen.setViewPort(float(window.swapchainExtent.width), float(window.swapchainExtent.height));

				// recreate off screen attachments
				for (auto &attach : offScreenAttachments) {
					attach.setFrameSize(window.swapchainExtent);
					attach.create(window, allocator, dpool, renderPass, screen_sampler.handle);
				}
			}

			framebuffers.clear();
			create_swapchain_framebuffers( window, renderPass.handle, framebuffers, offScreenAttachments, depthBufferView.handle); 
			
			recreateSwapchain = false;
			shouldGeneratePipeLine = true;
			// clear all pipelines
			pipelineMap.clear();
			continue;
		}
		
		// generate pipeline
		if (shouldGeneratePipeLine || modeChanged) {
			// check if there is a pipeline cache
			// also lazy loading here
			if (pipelineMap[sCurrentMode].empty()) {
				pipelineMap[sCurrentMode].emplace_back(
					model->
					bindPipeLine(basicPipeGen)
					.bindVertShader(baseVert)
					.bindFragShader(baseFrag)
					.addDescLayout(model->pbrTextureLayout.handle)
					.addDescLayout(model->metrialLayout())
					.generate(window, renderPass.handle)
				);
				if (sCurrentMode == cfg::Basic) {
					pipelineMap[cfg::Basic].emplace_back(
						offScreenAttachments[0]
						.bindPipeline(screenPipeGen)
						.bindVertShader(screenVert)
						.bindFragShader(screenFrag)
						.generate(window, renderPass.handle, 1)
					);
					pipelineMap[cfg::Basic].emplace_back(
						offScreenAttachments[1]
						.bindPipeline(screenPipeGen)
						.bindVertShader(screenVert)
						.bindFragShader(screenFrag)
						.generate(window, renderPass.handle, 2)
					);
				} else if (sCurrentMode == cfg::Tone) {
					pipelineMap[cfg::Tone].emplace_back(
						offScreenAttachments[0]
						.bindPipeline(screenPipeGen)
						.bindVertShader(screenVert)
						.bindFragShader(screenFrag)
						.generate(window, renderPass.handle, 1)
					);
					pipelineMap[cfg::Tone].emplace_back(
						offScreenAttachments[1]
						.bindPipeline(screenPipeGen)
						.bindVertShader(screenVert)
						.bindFragShader(toneFrag)
						.generate(window, renderPass.handle, 2)
					);
				} else if (sCurrentMode == cfg::Bloom) {
					pipelineMap[cfg::Bloom].emplace_back(
						offScreenAttachments[0]
						.bindPipeline(screenPipeGen)
						.addDescLayout(gaussSSBO.layout.handle)
						.addConstantRange(constants.m_range)
						.bindVertShader(screenVert)
						.bindFragShader(bloom_vFrag)
						.generate(window, renderPass.handle, 1)
					);
					pipelineMap[cfg::Bloom].emplace_back(
						offScreenAttachments[0]
						.bindPipeline(
						offScreenAttachments[1]
						.bindPipeline(screenPipeGen))
						.addDescLayout(gaussSSBO.layout.handle)
						.addConstantRange(constants.m_range)
						.bindVertShader(screenVert)
						.bindFragShader(bloom_hFrag)
						.generate(window, renderPass.handle, 2)
					);
				}
			}

			shouldGeneratePipeLine = false;
			modeChanged = false;
		}

		// acquire swapchain image.
		std::uint32_t imageIndex = 0;
		const auto acquireRes = vkAcquireNextImageKHR(
			window.device, 
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);
		if( VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes ) {
			recreateSwapchain = true;
			continue;
		}
		if (VK_SUCCESS != acquireRes) {
			throw lut::Error("Unable to acquire next swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}
		// wait for command buffer to be available
		// make sure that the command buffer is no longer in use
		assert( std::size_t(imageIndex) < cbfences.size());
		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, 
			std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res) {
				throw lut::Error( "Unable to wait for command buffer fence %u\n"
					"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}
		if( auto const res = vkResetFences( window.device, 1, &cbfences[imageIndex].handle ); 
			VK_SUCCESS != res ) {
				throw lut::Error( "Unable to reset command buffer fence %u\n"
					"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		// record and submit commands
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		auto BeginPipeline = [&](VkCommandBuffer cmdBuffer, const RenderPipeLine &pipe) {
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipe.handle);
			vkCmdBindDescriptorSets(cmdBuffer, 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				pipe.layout.handle, 0,
				1, 
				&sceneUBO.set, 
				0, nullptr
			);
			vkCmdBindDescriptorSets(cmdBuffer, 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				pipe.layout.handle, 1,
				1, 
				&lightUBO.set, 
				0, nullptr
			);
		};

		auto BindingMatSet = [](
			const RenderPipeLine &pipe, 
			VkCommandBuffer cmdBuffer, 
			VkDescriptorSet mat_set,
			unsigned int set_index) {
			vkCmdBindDescriptorSets(cmdBuffer, 
				VK_PIPELINE_BIND_POINT_GRAPHICS, 
				pipe.layout.handle, set_index,
				1, 
				&mat_set, 
				0, nullptr
			);
		};
		// clear query pool
		record_commands(
			cbuffers[imageIndex], 
			renderPass.handle, 
			framebuffers[imageIndex].handle,
			window.swapchainExtent,
			[&](VkCommandBuffer cmdBuffer) {
				// upload ubo
				sceneUBO.upload(cmdBuffer, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
				lightUBO.upload(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
				gaussSSBO.upload(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
				vkCmdResetQueryPool(cmdBuffer, queryPool.handle, 0, 4);
			},
			// real draw task
			[&](VkCommandBuffer cmdBuffer) {
				auto& pipe = pipelineMap[sCurrentMode][0];
				auto& screenPipe = pipelineMap[sCurrentMode][1];
				auto& screenPipe2 = pipelineMap[sCurrentMode][2];
				BeginPipeline(cmdBuffer, pipe);
				model->onDraw(cmdBuffer, [&](VkDescriptorSet tex_set, VkDescriptorSet mat_set) {
					BindingMatSet(pipe, cmdBuffer, tex_set, 2);
					BindingMatSet(pipe, cmdBuffer, mat_set, 3);
				});

				// next subpass 1
				vkCmdNextSubpass(cmdBuffer, VK_SUBPASS_CONTENTS_INLINE);
				// bind screen pipe
				vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, screenPipe.pipe.handle);
				// bind offscreen texture 
				BindingMatSet(screenPipe, cmdBuffer, offScreenAttachments[0].attachments_set, 0);
				if (sCurrentMode == cfg::Bloom) {
					BindingMatSet(screenPipe, cmdBuffer, gaussSSBO.set, 1);
					constants.bind(cmdBuffer, screenPipe.layout.handle, &aConstants);
				}
				
				vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, queryPool.handle, 0);
				// draw screen quad
				vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
				vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, queryPool.handle, 1);
				// next subpass 2
				vkCmdNextSubpass(cmdBuffer, VK_SUBPASS_CONTENTS_INLINE);
				// bind screen pipe
				vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, screenPipe2.pipe.handle);
				// bind offscreen texture 
				BindingMatSet(screenPipe2, cmdBuffer, offScreenAttachments[1].attachments_set, 0);
				if (sCurrentMode == cfg::Bloom) {
					BindingMatSet(screenPipe2, cmdBuffer, offScreenAttachments[0].attachments_set, 1);
					BindingMatSet(screenPipe2, cmdBuffer, gaussSSBO.set, 2);
					constants.bind(cmdBuffer, screenPipe2.layout.handle, &aConstants);
				}
				// draw screen quad
				vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, queryPool.handle, 2);
				// draw screen quad
				vkCmdDraw(cmdBuffer, 6, 1, 0, 0);
				vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, queryPool.handle, 3);
			}
		);

		submit_commands(window, 
			cbuffers[imageIndex],
			cbfences[imageIndex].handle, 
			imageAvailable.handle, 
			renderFinished.handle
		);

		// present rendered images.
		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);
		std::printf("----------------------------------------\n");
		std::printf("FPS: %.3f \n", 1.f / duration);
		uint64_t times[4];
		vkGetQueryPoolResults(window.device, queryPool.handle, 0, 4, 4 * sizeof(uint64_t), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		std::printf("Subpass 1 cost: %f ms\n", (times[1] - times[0]) * 1e-6);
		std::printf("Subpass 2 cost: %f ms\n", (times[3] - times[2]) * 1e-6);
	}
	
	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void update_scene_uniforms(
		glsl::SceneUniform& aSceneUniforms, 
		FpsController& aController,
		std::uint32_t aFramebufferWidth, 
		std::uint32_t aFramebufferHeight )
	{
		
		// initialize SceneUniform members
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);
		aSceneUniforms.P = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
	
		aSceneUniforms.P[1][1] *= -1.f; // mirror Y axis

		aSceneUniforms.camPos = glm::vec4(aController.m_position, 1.f);
		
		// move camera to center
		glm::mat4 V = glm::translate(-aController.m_position);
		// ratate the camera back to the front direction
		V = glm::eulerAngleXYZ(
			glm::radians(-aController.m_rotation.x), 
			glm::radians(-aController.m_rotation.y), 0.f) * V;
		
		// V is just move object to camera space
		aSceneUniforms.V = V;
		// no model change here
		aSceneUniforms.M = glm::mat4(1.f);
	}
}

namespace
{
	lut::RenderPass create_render_pass( lut::VulkanWindow const& aWindow )
	{
		VkAttachmentDescription attachments[4]{};
		// // color attachment texture 1
		attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		// color attachment texture 2
		attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		// color attachment
		attachments[2].format = aWindow.swapchainFormat;
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		// depth attachment
		attachments[3].format = cfg::kDepthFormat;
		attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; 
		attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// model pass
		VkAttachmentReference subpass1_Attachments[1]{};
		subpass1_Attachments[0].attachment = 0; 
		subpass1_Attachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// screen pass 1
		VkAttachmentReference subpass2_Attachments[1]{};
		subpass2_Attachments[0].attachment = 1; 
		subpass2_Attachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// screen pass 2 present
		VkAttachmentReference subpass3_Attachments[1]{};
		subpass3_Attachments[0].attachment = 2; 
		subpass3_Attachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpass2_inputAttachments[1]{};
		subpass2_inputAttachments[0].attachment = 0; 
		subpass2_inputAttachments[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference subpass3_inputAttachments[2]{};
		subpass3_inputAttachments[0].attachment = 0; 
		subpass3_inputAttachments[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		subpass3_inputAttachments[1].attachment = 1; 
		subpass3_inputAttachments[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference depthAttachment{}; 
		depthAttachment.attachment = 3;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		
		VkSubpassDescription subpasses[3]{}; 
		// RENDER PASS 1, mesh pass
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1; 
		subpasses[0].pColorAttachments = subpass1_Attachments; 
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		// RENDER PASS 2, screen pass 1
		subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[1].colorAttachmentCount = 1; 
		subpasses[1].pColorAttachments = subpass2_Attachments; 
		subpasses[1].inputAttachmentCount = 1;
		subpasses[1].pInputAttachments = subpass2_inputAttachments;

		// RENDER PASS 2, screen pass 2
		subpasses[2].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[2].colorAttachmentCount = 1; 
		subpasses[2].pColorAttachments = subpass3_Attachments; 
		subpasses[2].inputAttachmentCount = 2;
		subpasses[2].pInputAttachments = subpass3_inputAttachments;

		// create dependency
		VkSubpassDependency dependencies[2]{};
		// wait for the mesh pass to finish before starting the screen pass
		dependencies[0].srcSubpass = 0;
		dependencies[0].dstSubpass = 1;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 1;
		dependencies[1].dstSubpass = 2;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = sizeof(attachments) / sizeof(VkAttachmentDescription); 
		passInfo.pAttachments = attachments; 
		passInfo.subpassCount = sizeof(subpasses) / sizeof(VkSubpassDescription);
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = sizeof(dependencies) / sizeof(VkSubpassDependency);
		passInfo.pDependencies = dependencies;

		VkRenderPass rpass = VK_NULL_HANDLE; 
		if( auto const res = vkCreateRenderPass( aWindow.device, &passInfo, nullptr, &rpass); 
			VK_SUCCESS != res ) {
			throw lut::Error( "Unable to create render pass\n" 
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}
		return lut::RenderPass( aWindow.device, rpass );
	}

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const& aWindow, 
		VkRenderPass aRenderPass, 
		std::vector<lut::Framebuffer>& aFramebuffers,
		std::vector<VkColorAttachments> const& aColorAttachments,
		VkImageView aDepthView)
	{
		assert( aFramebuffers.empty() );

		for (std::uint32_t i = 0; i < aWindow.swapViews.size(); i ++) {
			std::vector<VkImageView> attachments;
			attachments.reserve(4);
			for (auto const& aColorAttachment : aColorAttachments)
				aColorAttachment.bindAttachments( attachments );
			attachments.emplace_back( aWindow.swapViews[i] );
			attachments.emplace_back( aDepthView);

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; 
			fbInfo.renderPass = aRenderPass; 
			fbInfo.attachmentCount = uint32_t(attachments.size()); 
			fbInfo.pAttachments = attachments.data(); 
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1; 

			VkFramebuffer fb = VK_NULL_HANDLE; 
			if( auto const res = vkCreateFramebuffer( aWindow.device, &fbInfo, nullptr, &fb); 
				VK_SUCCESS != res ) {
				throw lut::Error( "Unable to create framebuffer for swap chain image %zu\n"
					"vkCreateFramebuffer() returned %s", i,lut::to_string(res).c_str());
			}
			aFramebuffers.emplace_back( lut::Framebuffer( aWindow.device, fb ) );
		}

		assert( aWindow.swapViews.size() == aFramebuffers.size() );
	}

	lut::QueryPool create_query_pool(lut::VulkanWindow const& aWindow) 
	{
		VkQueryPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		poolInfo.queryCount = 4;
		poolInfo.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;

		VkQueryPool pool = VK_NULL_HANDLE;

		if( auto const res = vkCreateQueryPool( aWindow.device, &poolInfo, nullptr, &pool); 
			VK_SUCCESS != res ) {
			throw lut::Error( "Unable to create query pool\n" 
				"vkCreateQueryPool() returned %s", lut::to_string(res).c_str());
		}
		return lut::QueryPool( aWindow.device, pool );
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer( 
		lut::VulkanWindow const& aWindow,
		lut::Allocator const& aAllocator) 
	{ 
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D; 
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1; 
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; 
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; 
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE; 
		if( auto const res = vmaCreateImage( aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr ); 
			VK_SUCCESS != res )
		{
			throw lut::Error( "Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage( aAllocator.allocator, image, allocation );

		// Create the image view
		VkImageViewCreateInfo viewInfo{}; 
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image; 
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat; 
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1, 0, 1};
		
		VkImageView view = VK_NULL_HANDLE;

		if( auto const res = vkCreateImageView( aWindow.device, &viewInfo, nullptr, &view );
			VK_SUCCESS != res )
		{
			throw lut::Error( "Unable to create image view\n" 
				"vkCreateImageView() returned %s", lut::to_string(res).c_str() );
		}
		return { std::move(depthImage), lut::ImageView( aWindow.device, view ) };
	}

	void record_commands(
		VkCommandBuffer aCmdBuff, 
		VkRenderPass aRenderPass, 
		VkFramebuffer aFramebuffer, 
		VkExtent2D const& aImageExtent,
		std::function<void(VkCommandBuffer)> const& beforeRenderTask,
		std::function<void(VkCommandBuffer)> const& renderTask)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;
		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo); VK_SUCCESS != res) {
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// before render task
		beforeRenderTask(aCmdBuff);

		// Begin render pass
		VkClearValue clearValues[4]{};
		for (int i = 0; i < 3; ++i) {
			clearValues[i].color.float32[0] = 0.1f; // Clear to a dark gray background.
			clearValues[i].color.float32[1] = 0.1f; // Clear to a dark gray background.
			clearValues[i].color.float32[2] = 0.1f; // Clear to a dark gray background.
			clearValues[i].color.float32[3] = 1.f; // Clear to a dark gray background.
		}
		
		clearValues[3].depthStencil.depth = 1.f; // clear depth as 1.0f

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = aImageExtent;
		passInfo.clearValueCount = sizeof(clearValues) / sizeof(VkClearValue);  
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);
		
		renderTask(aCmdBuff);

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);

		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res) {
			throw lut::Error("Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}

	void submit_commands( lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore )
	{
		VkPipelineStageFlags waitPipelineStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;


		if( auto const res = vkQueueSubmit( aContext.graphicsQueue, 1, &submitInfo, aFence )
			; VK_SUCCESS != res ) {
			throw lut::Error( "Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void present_results( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
	{
		// present the results
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1; 
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1; 
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR( aPresentQueue, &presentInfo );

		if( VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes ) {
			aNeedToRecreateSwapchain = true; 
		}
		else if( VK_SUCCESS != presentRes ) {
			throw lut::Error( "Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}

	// helper
}


//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
