module LOSChecker

import ..Types: Position
import ..Utils: pos_distance, operate_pos

using Infinity

function check_los(obstacle_map::Array, 
                   scale_factor::Float64, 
                   source::Position, 
                   target::Position, 
                   range::Float64)

    if pos_distance(source, target) > range 
        return false
    end

    scaled_source = operate_pos(source, scale_factor, /)
    scaled_target = operate_pos(target, scale_factor, /)

    current_pixel::Array{Int64} = [ceil(scaled_source.y), ceil(scaled_source.x)]
    current_position = [Float64(p) for p in current_pixel]
    target_pixel::Array{Int64} = [ceil(scaled_target.y), ceil(scaled_target.x)]

    pixel_distance = (target_pixel[1] - current_pixel[1], target_pixel[2] - current_pixel[2])

    function get_bound(x, dir)
        if sign(dir) == 1
            return floor(x) + 1
        else
            return ceil(x) - 1
        end
    end

    while true
        if obstacle_map[current_pixel[1], current_pixel[2]] < 0.01
            return false
        end
        if current_pixel == target_pixel
            return true
        end

        m_ax_1 = (get_bound(current_position[1], pixel_distance[1]) - current_position[1]) / (pixel_distance[1])
        m_ax_2 = (get_bound(current_position[2], pixel_distance[2]) - current_position[2]) / (pixel_distance[2])

        if isnan(m_ax_1)
            m_ax_1 = ∞
        end
        if isnan(m_ax_2)
            m_ax_2 = ∞
        end

        if abs(m_ax_1) < abs(m_ax_2)
            current_pixel[1] += sign(pixel_distance[1])
            m = m_ax_1
        else 
            current_pixel[2] += sign(pixel_distance[2])
            m = m_ax_2
        end

        current_position = [current_position[1] + m*pixel_distance[1], current_position[2]+m*pixel_distance[2]]

    end

end

end