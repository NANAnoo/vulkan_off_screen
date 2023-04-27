#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <volk/volk.h>

#include "../labutils/vkutil.hpp"
#include "../labutils/error.hpp"
#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_context.hpp"
#include "../labutils/vkobject.hpp"

#include <glm/glm.hpp>

#include <string>
#include <unordered_map>

// As the pipe need so many informations,
// there is a need for generator design pattern

class RenderPipeLine {
    public:
    labutils::Pipeline pipe;
    labutils::PipelineLayout layout;
};

class PipeLineGenerator {
public:
    PipeLineGenerator() = default;

    // its okay to copy this
    PipeLineGenerator(const PipeLineGenerator &other) = default;
    PipeLineGenerator& operator=(const PipeLineGenerator &other) = default;

    // it's ok to move too
    PipeLineGenerator(PipeLineGenerator &&other) = default;
    PipeLineGenerator& operator=(PipeLineGenerator &&other) = default;

    PipeLineGenerator &bindVertShader(labutils::ShaderModule &vert) {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vert.handle;
        stage.pName = "main";
        m_shader_stages.emplace_back(stage);
        return *this;
    }

    PipeLineGenerator &bindFragShader(labutils::ShaderModule &frag) {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage.module = frag.handle;
        stage.pName = "main";
        m_shader_stages.emplace_back(stage);
        return *this;
    }

    PipeLineGenerator &bindGeomShader(labutils::ShaderModule &geom) {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
        stage.module = geom.handle;
        stage.pName = "main";
        m_shader_stages.emplace_back(stage);
        return *this;
    }

    PipeLineGenerator &addDescLayout(const VkDescriptorSetLayout &layout) {
        layouts.push_back(layout);
        return *this;
    }

    PipeLineGenerator &addConstantRange(const VkPushConstantRange &range) {
        constants.push_back(range);
        return *this;
    }

    PipeLineGenerator &addVertexInfo(
        std::uint32_t binding,
        std::uint32_t location,
        std::uint32_t stride,
        VkFormat format,
        std::uint32_t offset = 0) 
    {
        m_vertex_iuputs.push_back({
            binding,
            location,
            stride,
            format,
            offset
        });
        return *this;
    }

    PipeLineGenerator &setViewPort(float w, float h) {
        m_width = w;
        m_height = h;
        return *this;
    }

    PipeLineGenerator &setCullMode(VkCullModeFlagBits cullMode) {
        m_cull_mode = cullMode;
        return *this;
    }

    PipeLineGenerator &setPolyGonMode(VkPolygonMode polyGonMode) {
        m_polygon_mode = polyGonMode;
        return *this;
    }

    PipeLineGenerator &setRenderMode(VkPrimitiveTopology mode) {
        m_topology = mode;
        return *this;
    }

    PipeLineGenerator &enableDepthTest(bool enable) {
        m_depth_test_enabled = enable;
        return *this;
    }

    PipeLineGenerator &enableBlend(bool enable) {
        m_blend_enabled = enable;
        return *this;
    }

    PipeLineGenerator &setScissor(glm::vec4 scissor) {
        m_scissor = scissor;
        return *this;
    }

    RenderPipeLine generate(
        labutils::VulkanContext const& aContext, 
        VkRenderPass aRenderPass,
        std::uint32_t whichSubPass = 0)
    {
        // create layout first
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = static_cast<std::uint32_t>(layouts.size());
        layoutInfo.pSetLayouts = layouts.data();
        layoutInfo.pushConstantRangeCount = static_cast<std::uint32_t>(constants.size());
        layoutInfo.pPushConstantRanges = constants.data();

        VkPipelineLayout layout = VK_NULL_HANDLE;
        if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res) {
			throw labutils::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", labutils::to_string(res).c_str());
		}

        // create vertex inputs description
        VkPipelineVertexInputStateCreateInfo inputInfo{};
        inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        std::vector<VkVertexInputBindingDescription> vertexInputs;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
        for (auto &aInput : m_vertex_iuputs) {
            VkVertexInputBindingDescription input{};
            input.binding = aInput.binding;
            input.stride = aInput.stride;
            input.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            vertexInputs.emplace_back(input);

            VkVertexInputAttributeDescription attribute{};
            attribute.binding = aInput.binding;
            attribute.location = aInput.location;
            attribute.format = aInput.format;
            attribute.offset = aInput.offset;
            vertexAttributes.push_back(attribute);
        }
        inputInfo.vertexBindingDescriptionCount = static_cast<std::uint32_t>(vertexInputs.size()); // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = vertexInputs.data(); 
		inputInfo.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vertexAttributes.size()); // number of vertexAttributes above
        inputInfo.pVertexAttributeDescriptions = vertexAttributes.data();

        // assembly information
        VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
        assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = m_topology;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

        // define viewport state
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = m_width;
		viewport.height = m_height;
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ int(m_scissor.x * m_width), int(m_scissor.y * m_height) };
		scissor.extent = {
            std::uint32_t(m_scissor.z * m_width),
            std::uint32_t(m_scissor.w * m_height)
        };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

        // define rasterization state
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = m_polygon_mode;
		rasterInfo.cullMode = m_cull_mode;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required.

        // define depth test state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = m_depth_test_enabled ? VK_TRUE : VK_FALSE;
        depthInfo.depthWriteEnable = m_depth_test_enabled ? VK_TRUE : VK_FALSE;
        depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthInfo.minDepthBounds = 0.f; 
        depthInfo.maxDepthBounds = 1.f;

        // Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // define color blend state
        // TODO blend type
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = m_blend_enabled ? VK_TRUE : VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT|
										VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

        // Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = static_cast<std::uint32_t>(m_shader_stages.size());
		pipeInfo.pStages = m_shader_stages.data();

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states

		pipeInfo.layout = layout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = whichSubPass; // first subpass of aRenderPass

        VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(
                aContext.device,
                VK_NULL_HANDLE, 
                1, &pipeInfo, 
                nullptr, &pipe);
			VK_SUCCESS != res) {
			throw labutils::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", labutils::to_string(res).c_str());
		}

        return {
            labutils::Pipeline(aContext.device, pipe),
            labutils::PipelineLayout(aContext.device, layout)
        };
    }

private:
    // descripyor layout
    std::vector<VkDescriptorSetLayout> layouts;
    // shaders
    std::vector<VkPipelineShaderStageCreateInfo> m_shader_stages;
    std::vector<VkPushConstantRange> constants;
    struct VertexInfo {
        std::uint32_t binding;
        std::uint32_t location;
        std::uint32_t stride;
        VkFormat format;
        std::uint32_t offset;
    };

    // input assemblies
    std::vector<VertexInfo> m_vertex_iuputs;
    VkPrimitiveTopology m_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    
    // view port
    float m_width = 0;
    float m_height = 0;
    // x, y, w, h
    glm::vec4 m_scissor = {0, 0, 1, 1};

    // rasterization state
    VkCullModeFlagBits m_cull_mode = VK_CULL_MODE_NONE;
    VkPolygonMode  m_polygon_mode = VK_POLYGON_MODE_FILL;

    // depth
    bool m_depth_test_enabled = false;

    // TODO blend state
    bool m_blend_enabled = false;
};