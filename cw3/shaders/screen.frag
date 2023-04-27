#version 460

layout( location = 0 ) in vec2 v2f_tc;

layout( location = 0 ) out vec4 oColor;

layout( set = 0, binding = 0 ) uniform sampler2D screenTexture;


void main()
{
    oColor = texture( screenTexture, v2f_tc ).rgba;
}