#version 460

layout( location = 0 ) in vec2 v2f_tc;

layout( location = 0 ) out vec4 oColor;

layout( set = 0, binding = 0 ) uniform sampler2D screenTexture;


void main()
{
    vec4 color = texture( screenTexture, v2f_tc ).rgba;
    oColor = color / (1.0 + color);
}