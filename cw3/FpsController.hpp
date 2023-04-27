#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdio>
#include <glm/common.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <unordered_map>
#include <unordered_set>



class FpsController {
public:
    enum MoveType {
        MoveUp = 1u << 0,
        MoveDown = 1u << 1,
        MoveForward = 1u << 2,
        MoveBackward = 1u << 3,
        MoveLeft = 1u << 4,
        MoveRight = 1u << 5,
        MoveSpeedUp = 1u << 6,
        MoveSpeedDown = 1u << 7
    };

    FpsController(glm::vec3 pos,
        glm::vec3 rotation, 
        float speed = 1, 
        float scaleFactor = 2,
        glm::vec3 up = {0, 1, 0}
    ):
        m_rotation(rotation), 
        m_position(pos), 
        m_up(glm::normalize(up)),
        m_speed(speed),
        m_scale(scaleFactor < 1 ? 1 : scaleFactor)
    {
        // default map:
        keyMap[GLFW_KEY_W] = MoveForward;
        keyMap[GLFW_KEY_S] = MoveBackward;
        keyMap[GLFW_KEY_A] = MoveLeft;
        keyMap[GLFW_KEY_D] = MoveRight;
        keyMap[GLFW_KEY_Q] = MoveDown;
        keyMap[GLFW_KEY_E] = MoveUp;
        keyMap[GLFW_KEY_LEFT_SHIFT] = MoveSpeedUp;
        keyMap[GLFW_KEY_LEFT_CONTROL] = MoveSpeedDown;
        updateFront();
    }

    void bindKey(int keycode, MoveType move) {
        keyMap[keycode] = move;
    }

    void onKeyPress(int keycode) {
        if (keyMap.find(keycode) != keyMap.end()) {
            // record down key
            downKeys |= (unsigned int)(keyMap[keycode]);
        }
    }

    void onKeyUp(int keycode) {
        // erase down key
        downKeys &= ~keyMap[keycode];
    }

    void changeSpeed(float scale) {
        m_scale = scale;
    }

    void onCursorMove(float offset_x, float offset_y) {
        m_rotation.x -= offset_y / 10.f;
        m_rotation.y -= offset_x / 10.f;
        // avoid lock
        m_rotation.x = std::max(-80.f, std::min(m_rotation.x, 80.f));
        updateFront();
    }

    void updateFront() {
        const static glm::vec4 front_base = { 0, 0, -1, 0 };
        m_front = glm::eulerAngleZYX(
            0.f,
            glm::radians(m_rotation.y),
            glm::radians(m_rotation.x))
            * front_base;
        m_front = glm::normalize(m_front);
    }

    void update(float duration) {
        glm::vec3 move_direction = {0, 0, 0};
        glm::vec3 left = glm::normalize(glm::cross(m_up, m_front));
        float scale = 1;
        if ((downKeys & MoveForward) > 0) 
            move_direction += m_front;
        if ((downKeys & MoveBackward) > 0) move_direction -= m_front;
        if ((downKeys & MoveLeft) > 0) move_direction += left;
        if ((downKeys & MoveRight) > 0) move_direction -= left;
        if ((downKeys & MoveUp) > 0) move_direction += m_up;
        if ((downKeys & MoveDown) > 0) move_direction -= m_up;
        if ((downKeys & MoveSpeedUp) > 0) scale *= m_scale;
        if ((downKeys & MoveSpeedDown) > 0) scale /= m_scale;

        m_position += duration * m_speed * scale * move_direction;
        
    }

    glm::vec3 m_rotation;
    glm::vec3 m_position;
    glm::vec3 m_up;
private:
    glm::vec3 m_front;
    
    float m_speed;
    float m_scale = 2;

    std::unordered_map<int, MoveType> keyMap;
    unsigned int downKeys = 0;
};