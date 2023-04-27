#version 460

const vec2 positions[6] = {
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
};

layout( location = 0 ) out vec2 v2f_tc;

void main()
{
    v2f_tc = positions[gl_VertexIndex];
    gl_Position = vec4(v2f_tc * 2 - 1, 0.0, 1.0);
}
