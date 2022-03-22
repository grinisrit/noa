/*****************************************************************************
 *   Copyright (c) 2022, Roland Grinis, GrinisRIT ltd.                       *
 *   (roland.grinis@grinisrit.com)                                           *
 *   All rights reserved.                                                    *
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/**
 * Implemented by: Oleg Mosiagin
 */

#pragma once

#include <noa/utils/common.hh>

#include <noa/3rdparty/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/TNL/Containers/StaticArray.h>

namespace noa::pms::trace {
    using namespace noa::TNL;
    using namespace noa::TNL::Containers;

    template <class DeviceType>
    class Tracer {
    private:
        using Mesh = Meshes::Mesh<Meshes::DefaultConfig<Meshes::Topologies::Tetrahedron>, DeviceType>;
        using Point = typename Mesh::PointType;
        using Real = typename Mesh::RealType;
        using Index = typename Mesh::GlobalIndexType;
        using LocalIndex = typename Mesh::LocalIndexType;

        __cuda_callable__
        static Real get_ray_plane_intersection(
                const Point &point0,
                const Point &point1,
                const Point &point2,
                const Point &origin,
                const Point &direction) {
            Point edge1 = point1 - point0;
            Point edge2 = point2 - point0;

            Point normal = VectorProduct(edge1, edge2);
            if (dot(point0 - origin, normal) < 0) {
                normal = -normal;
            }
            normal /= sqrt(dot(normal, normal));

            Real direction_projection = dot(normal, direction);

            if (direction_projection > 0) {
                Point to_triangle = point0 - origin;
                Real to_triangle_projection = dot(to_triangle, normal);

                return to_triangle_projection / direction_projection;
            }

            return -1;
        }

    public:
        struct Intersection {
            Index tetrahedron_global_index = -1;
            Real distance = -1;
        };

        __cuda_callable__
        static std::optional<Intersection> get_first_border_in_tetrahedron(
            const Mesh* mesh_pointer,
            const Index tetrahedron_global_index,
            const Point& origin,
            const Point& direction,
            Real epsilon) {
            TNL_ASSERT_EQ(Mesh::getMeshDimension(), 3, "wrong mesh dimension");
            const typename Mesh::Cell &tetrahedron = mesh_pointer->template getEntity<Mesh::getMeshDimension()>(tetrahedron_global_index);

            Real minimal_distance = std::numeric_limits<Real>::max();
            LocalIndex nearest_face_id = 0;
            Real second_minimal_distance = std::numeric_limits<Real>::max();

            for (LocalIndex face_id = 0; face_id < tetrahedron.template getSubentitiesCount<2>(); face_id++) {
                Index face_global_index = tetrahedron.template getSubentityIndex<2>(face_id);
                const auto &face = mesh_pointer->template getEntity<2>(face_global_index);

                TNL_ASSERT_EQ(face.template getSubentitiesCount<0>(), 3, "wrong number of vertices");
                Point points[3] = {};
                for (LocalIndex point_id = 0; point_id < 3; point_id++) {
                    Index point_global_index = face.template getSubentityIndex<0>(point_id);
                    const auto &point_entity = mesh_pointer->template getEntity<0>(point_global_index);
                    points[point_id] = point_entity.getPoint();
                }

                Real current_distance = get_ray_plane_intersection(points[0], points[1], points[2], origin, direction);

                if (current_distance > 0) {
                    if (current_distance < minimal_distance) {
                        second_minimal_distance = minimal_distance;
                        minimal_distance = current_distance;
                        nearest_face_id = face_id;
                    } else if (current_distance < second_minimal_distance) {
                        second_minimal_distance = current_distance;
                    }
                }
            }

            if (second_minimal_distance - minimal_distance < epsilon) {
                return {};
            }

            Intersection result;
            result.distance = minimal_distance;

            Index nearest_face_global_index = tetrahedron.template getSubentityIndex<2>(nearest_face_id);
            const auto &nearest_face = mesh_pointer->template getEntity<2>(nearest_face_global_index);

            if (nearest_face.template getSuperentitiesCount<3>() != 1) {
                TNL_ASSERT_EQ(nearest_face.template getSuperentitiesCount<3>(), 2, "wrong number of tetrahedrons");

                for (LocalIndex intersection_tetrahedron_id = 0; intersection_tetrahedron_id < 2; intersection_tetrahedron_id++) {
                    Index intersection_tetrahedron_global_index = nearest_face.template getSuperentityIndex<3>(intersection_tetrahedron_id);
                    if (intersection_tetrahedron_global_index != tetrahedron_global_index) {
                        result.tetrahedron_global_index = intersection_tetrahedron_global_index;
                        break;
                    }
                }
            }

            return result;
        }
    };
}
