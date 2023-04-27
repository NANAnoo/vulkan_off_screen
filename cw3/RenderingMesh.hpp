#pragma once

#include "RenderProgram.hpp"
#include "baked_model.hpp"
#include "VkVertex.hpp"
#include "VkUBO.hpp"

#include <functional>
#include <memory>

namespace {
    struct RenderingMesh {
        std::uint32_t materialId = 0;
        std::unique_ptr<VkVBO> positions = nullptr;
        std::unique_ptr<VkVBO> normals = nullptr;
        std::unique_ptr<VkVBO> texcoords = nullptr;

        std::unique_ptr<VkIBO> indices = nullptr;

        RenderingMesh() noexcept = default;
        // disable copy
        RenderingMesh(const RenderingMesh&) = delete;
        RenderingMesh operator=(const RenderingMesh&) = delete;

        // enable move
        RenderingMesh(RenderingMesh&& other) noexcept {
            positions = std::move(other.positions);
            normals = std::move(other.normals);
            texcoords = std::move(other.texcoords);
            indices = std::move(other.indices);
        }
        RenderingMesh& operator=(RenderingMesh&& other) noexcept {
            positions = std::move(other.positions);
            normals = std::move(other.normals);
            texcoords = std::move(other.texcoords);
            indices = std::move(other.indices);
            return *this;
        }
    };

    // material uniform buffer
    struct MaterialInfo {
        glm::vec3 baseColor;
        float roughness;
        glm::vec3 emissiveColor;
        float metalness;
    };

    class RenderingModel {
    public:
        RenderingModel() noexcept = default;
        // disable copy
        RenderingModel(const RenderingModel&) = delete;
        RenderingModel operator=(const RenderingModel&) = delete;

        // enable move
        RenderingModel(RenderingModel&& other) noexcept {
            meshes_by_material = std::move(other.meshes_by_material);
        }
        RenderingModel& operator=(RenderingModel&& other) noexcept {
            meshes_by_material = std::move(other.meshes_by_material);
            return *this;
        }

        void load(
            lut::VulkanContext const& aContext, 
            lut::Allocator const& aAllocator,
            lut::DescriptorPool const& dPool,
            const BakedModel& model) 
        {
            texInfos = model.textures;
            materials = model.materials;
            for (size_t i = 0; i < model.meshes.size(); ++i) {
                auto& mesh = model.meshes[i];
                auto renderingMesh = RenderingMesh();

                renderingMesh.materialId = mesh.materialId;

                renderingMesh.positions = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.positions.size() * sizeof(glm::vec3),
                    (void *)(mesh.positions.data())
                );
                renderingMesh.normals = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.normals.size() * sizeof(glm::vec3),
                    (void *)(mesh.normals.data())
                );
                renderingMesh.texcoords = std::make_unique<VkVBO>(
                    aContext, aAllocator,
                    mesh.texcoords.size() * sizeof(glm::vec2),
                    (void *)(mesh.texcoords.data())
                );
                renderingMesh.indices = std::make_unique<VkIBO>(
                    aContext, aAllocator,
                    mesh.indices.size() * sizeof(uint32_t),
                    mesh.indices.size(), 
                    (void *)(mesh.indices.data())
                );
                meshes_by_material[mesh.materialId].push_back(std::move(renderingMesh));
            }

            auto TexDescLayourHelper = [&](unsigned int tex_count) {
                std::vector<VkDescriptorSetLayoutBinding> bindings(tex_count);
                for (int i = 0; i < int(tex_count); i ++) {
                    bindings[i].binding = i; // bind texture sampler 0
                    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    bindings[i].descriptorCount = 1; 
                    bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_GEOMETRY_BIT; 
                }
                
                VkDescriptorSetLayoutCreateInfo layoutInfo{};
                layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                layoutInfo.bindingCount = uint32_t(bindings.size());
                layoutInfo.pBindings = bindings.data();

                VkDescriptorSetLayout layout = VK_NULL_HANDLE;
                if( auto const res = vkCreateDescriptorSetLayout( 
                        aContext.device, 
                        &layoutInfo,
                        nullptr, 
                        &layout ); 
                    VK_SUCCESS != res )
                {
                    throw lut::Error( "Unable to create descriptor set layout\n" 
                        "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
                }
                return lut::DescriptorSetLayout( aContext.device, layout ); 
            };

            // load textures
            std::vector<std::string> paths;
            for (auto& tex : texInfos) {
                paths.push_back(tex.path);
            }
            labutils::CommandPool tempPool = labutils::create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            auto res = labutils::loadTextures(aContext, tempPool.handle, aAllocator, paths);
            // create img view for each img
            for (auto& [path, img] : res) {
                auto texView = lut::create_image_view_texture2d(aContext, img.image, VK_FORMAT_R8G8B8A8_SRGB);
                PackedTexture tex{
                    std::move(img),
                    std::move(texView)
                };
                allTextures.insert({
                    path,
                    std::move(tex)
                });
            }
            // create descriptor set layout for textures
            pbrTextureLayout = TexDescLayourHelper(3);

            // create uniform buffer for materials
            for (auto &mat : model.materials) {
                auto uniformBuffer = std::make_shared<VkUBO<MaterialInfo>>(
                    aContext, aAllocator, dPool, VK_SHADER_STAGE_FRAGMENT_BIT
                );
                uniformBuffer->data = std::make_unique<MaterialInfo>();
                uniformBuffer->data->baseColor = mat.baseColor;
                uniformBuffer->data->emissiveColor = mat.emissiveColor;
                uniformBuffer->data->roughness = mat.roughness;
                uniformBuffer->data->metalness = mat.metalness;

                material_ubos.push_back(uniformBuffer);
            }
        }

        void upload(VkCommandBuffer uploadCmd) {
            for (auto& [_, meshes] : meshes_by_material) {
                for (auto& mesh : meshes) {
                    mesh.positions->upload(uploadCmd);
                    mesh.normals->upload(uploadCmd);
                    mesh.texcoords->upload(uploadCmd);
                    mesh.indices->upload(uploadCmd);
                }
            }
            // upload uniform buffers
            for (auto& ubo : material_ubos) {
                ubo->upload(uploadCmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            }
        }

        PipeLineGenerator bindPipeLine(PipeLineGenerator aGenerator) {
            aGenerator
            // positions
            .addVertexInfo(0, 0, sizeof(float) * 3, VK_FORMAT_R32G32B32_SFLOAT)
            // normals
            .addVertexInfo(1, 1, sizeof(float) * 3, VK_FORMAT_R32G32B32_SFLOAT)
            // texcoords
            .addVertexInfo(2, 2, sizeof(float) * 2, VK_FORMAT_R32G32_SFLOAT);
            return aGenerator;
        }

        void onDraw(VkCommandBuffer cmd, const std::function<void(VkDescriptorSet tex_set, VkDescriptorSet mat_set)> &aSetCallback) {
            for (auto& [mat_id, meshes] : meshes_by_material) {
                aSetCallback(tex_sets[mat_id], material_ubos[mat_id]->set);
                for (auto& mesh : meshes) {
                    VkBuffer buffers[3] = { mesh.positions->get(), mesh.normals->get(), mesh.texcoords->get()};
                    VkDeviceSize offsets[3] = { 0, 0, 0};
                    vkCmdBindVertexBuffers(cmd, 0, 3, buffers, offsets);
                    mesh.indices->bind(cmd);
                    mesh.indices->draw(cmd);
                }
            }
        }

        void createTextureSetsWith(
            lut::VulkanContext const& aContext,
            VkDescriptorPool dPool,
            VkSampler aSampler)
        {
            // create material descriptor sets
            std::vector<VkDescriptorImageInfo> imgInfos;
            std::vector<VkWriteDescriptorSet> texWriteSets;
            
            // have to reserve a enough space
            // since when vector increase
            // stl will move the old array buffer to the new buffer
            imgInfos.reserve(texInfos.size());
            // load & upload all textures
            for (auto &info :texInfos) {
                auto &tex = allTextures[info.path];
                imgInfos.push_back({
                    aSampler,
                    tex.view.handle,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                });
            }

            auto CreateWriteDesc = [&](VkDescriptorSet set, VkDescriptorImageInfo *imgInfo, unsigned int count) {
                VkWriteDescriptorSet desc{};
                desc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                desc.dstSet = set; 
                desc.dstBinding = 0; //
                desc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                desc.descriptorCount = count; 
                desc.pImageInfo = imgInfo; 
                texWriteSets.emplace_back(desc);
            };

            std::vector<std::vector<VkDescriptorImageInfo>> mat_imgInfos(materials.size());
            for (unsigned int i = 0; i < materials.size(); i ++) {
                auto &mat = materials[i];
                auto &mat_imgInfo = mat_imgInfos[i];
                mat_imgInfo.reserve(3);
                for (auto &idx : {
                    mat.baseColorTextureId, 
                    mat.metalnessTextureId, 
                    mat.roughnessTextureId,
                }) {
                    mat_imgInfo.emplace_back(imgInfos[idx]);
                }
                auto basic_set = lut::alloc_desc_set(aContext, dPool, pbrTextureLayout.handle);
                tex_sets.push_back(basic_set);
                CreateWriteDesc(basic_set, mat_imgInfo.data(), 3);
            }

            // update descriptor set
            vkUpdateDescriptorSets( aContext.device, static_cast<std::uint32_t>(texWriteSets.size()), texWriteSets.data(), 0, nullptr );
        }

        static void uploadScope(lut::VulkanContext const& aContext, const std::function<void(VkCommandBuffer)> &cb) {
            labutils::CommandPool tempPool = labutils::create_command_pool(aContext, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            // create upload cmd buffer, fence
            labutils::Fence uploadComplete = labutils::create_fence(aContext);
            VkCommandBuffer uploadCmd = labutils::alloc_command_buffer(aContext, tempPool.handle);

            // begin cmd
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;

            if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
                VK_SUCCESS != res) {
                throw lut::Error("Beginning command buffer recording\n"
                    "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
            }

            // record upload cmd
            cb(uploadCmd);

            // end cmd
            if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res) {
                throw lut::Error("Ending command buffer recording\n"
                    "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
            }
            // submit upload cmd
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &uploadCmd;

            if (auto const res = vkQueueSubmit(
                aContext.graphicsQueue, 
                1, &submitInfo, 
                uploadComplete.handle);
                VK_SUCCESS != res) {
                throw lut::Error( "Submitting commands\n" 
                    "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
            }

            // Wait for commands to finish before we destroy the temporary resources
            if( auto const res = vkWaitForFences( aContext.device, 1, &uploadComplete.handle,
                VK_TRUE, std::numeric_limits<std::uint64_t>::max() ); VK_SUCCESS != res ) {
                throw lut::Error( "Waiting for upload to complete\n" 
                    "vkWaitForFences() returned %s", lut::to_string(res).c_str());
            }
            vkFreeCommandBuffers( aContext.device, tempPool.handle, 1, &uploadCmd );
        }

        lut::DescriptorSetLayout pbrTextureLayout; // base color + metalness + roughness

        VkDescriptorSetLayout metrialLayout() {
            return material_ubos[0]->layout.handle;
        }

    private:
    	struct PackedTexture {
            labutils::Image img{};
            labutils::ImageView view{};
        };

        // texture data map
        std::unordered_map<std::string, PackedTexture> allTextures;

        std::vector<std::shared_ptr<VkUBO<MaterialInfo>>> material_ubos; 

        std::vector<VkDescriptorSet> tex_sets;

        std::vector<BakedTextureInfo> texInfos;
        std::vector<BakedMaterialInfo> materials;
        std::unordered_map<unsigned int, std::vector<RenderingMesh>> meshes_by_material;
    };
}