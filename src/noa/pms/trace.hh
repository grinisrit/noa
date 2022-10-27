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

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticArray.h>

namespace noa::pms::trace {
    using namespace noa::TNL;
    using namespace noa::TNL::Containers;

    template <class DeviceType, typename Real = float, typename Index = long int, typename LocalIndex = short int>
    class Tracer {
    private:
        using CellTopology = Meshes::Topologies::Tetrahedron;
        using MeshConfig = Meshes::DefaultConfig<CellTopology, CellTopology::dimension, Real, Index, LocalIndex>;
        using Mesh = Meshes::Mesh<MeshConfig, DeviceType>;
        using Point = typename Mesh::PointType;

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

        __cuda_callable__
        static bool check_points_in_one_side(
                const Point &plane_point0,
                const Point &plane_point1,
                const Point &plane_point2,
                const Point &point0,
                const Point &point1) {
            Point edge0 = plane_point1 - plane_point0;
            Point edge1 = plane_point2 - plane_point0;

            Point normal = VectorProduct(edge0, edge1);

            Real product0 = dot(point0 - plane_point0, normal);
            Real product1 = dot(point1 - plane_point0, normal);

            return product0 * product1 >= 0;
        }

        __cuda_callable__
        static bool check_point_in_tetrahedron(
                const Mesh *mesh_pointer,
                const Index tetrahedron_global_index,
                const Point &point) {
            const typename Mesh::Cell &tetrahedron = mesh_pointer->template getEntity<Mesh::getMeshDimension()>(
                    tetrahedron_global_index);

            TNL_ASSERT_EQ(tetrahedron.template getSubentitiesCount<0>(), 4, "wrong number of vertices");
            Point points[4] = {};
            for (LocalIndex point_id = 0; point_id < 4; point_id++) {
                Index point_global_index = tetrahedron.template getSubentityIndex<0>(point_id);
                const auto &point_entity = mesh_pointer->template getEntity<0>(point_global_index);
                points[point_id] = point_entity.getPoint();
            }

            bool check0 = check_points_in_one_side(points[0], points[1], points[2], points[3], point);
            bool check1 = check_points_in_one_side(points[1], points[2], points[3], points[0], point);
            bool check2 = check_points_in_one_side(points[0], points[2], points[3], points[1], point);
            bool check3 = check_points_in_one_side(points[0], points[1], points[3], points[2], point);

            return check0 && check1 && check2 && check3;
        }

    public:
        /// Intersection structure
        struct Intersection {
            /// Index of the first triangle on the ray
            Index nearest_face_global_index = -1;

            /// Distance to first triangle from ray origin
            Real distance = -1;

            /// Flag of intersection with a triangle:
            /// True -- if distance between first triangle on the ray and second triangle less than epsilon,
            /// False -- if otherwise
            bool is_intersection_with_triangle = true;
        };

        /// Get next tetrahedron after intersection
        /// \param mesh_pointer Pointer to device mesh
        /// \param intersection Intersection structure from get_first_border_in_tetrahedron function
        /// \param origin New ray origin (intersection point)
        /// \param direction New ray direction
        /// \param ray_offset Offset along ray for check if point (= origin + direction * ray_offset) is inside of a tetrahedron
        /// \return {Next tetrahedron in ray} -- if success, {} - if otherwise (tetrahedron not found)
        __cuda_callable__
        static std::optional<Index> get_next_tetrahedron(
                const Mesh *mesh_pointer,
                const Intersection &intersection,
                const Point &origin,
                const Point &direction,
                Real ray_offset) {
            Point point_with_offset = origin + direction * ray_offset;

            const auto &face = mesh_pointer->template getEntity<2>(intersection.nearest_face_global_index);
            for (LocalIndex tetrahedron_id = 0;
                 tetrahedron_id < face.template getSuperentitiesCount<3>(); tetrahedron_id++) {
                Index tetrahedron_global_index = face.template getSuperentityIndex<3>(tetrahedron_id);
                if (check_point_in_tetrahedron(mesh_pointer, tetrahedron_global_index, point_with_offset)) {
                    return tetrahedron_global_index;
                }
            }

            if (intersection.is_intersection_with_triangle) {
                return {};
            }

            for (LocalIndex point_id = 0; point_id < face.template getSubentitiesCount<0>(); point_id++) {
                Index point_global_index = face.template getSubentityIndex<0>(point_id);
                const auto &point = mesh_pointer->template getEntity<0>(point_global_index);
                for (LocalIndex tetrahedron_id = 0;
                     tetrahedron_id < point.template getSuperentitiesCount<3>(); tetrahedron_id++) {
                    Index tetrahedron_global_index = point.template getSuperentityIndex<3>(tetrahedron_id);
                    if (check_point_in_tetrahedron(mesh_pointer, tetrahedron_global_index, point_with_offset)) {
                        return tetrahedron_global_index;
                    }
                }
            }

            return {};
        }

        /// Calculate the triangle the ray hits
        /// \param mesh_pointer Pointer to device mesh
        /// \param tetrahedron_global_index Current tetrahedron global index in mesh (origin located here)
        /// \param origin Ray origin
        /// \param direction Ray direction
        /// \param epsilon Threshold distinguishing between intersections between a triangle and a line segment
        ///                (The distance between the first intersection with the plane and the second is checked)
        /// \return Intersection structure: triangle global index, distance from origin, (triangle)/(line or point) flag (see Intersection)
        __cuda_callable__
        static Intersection get_first_border_in_tetrahedron(
                const Mesh *mesh_pointer,
                const Index tetrahedron_global_index,
                const Point &origin,
                const Point &direction,
                Real epsilon) {
            TNL_ASSERT_EQ(Mesh::getMeshDimension(), 3, "wrong mesh dimension");
            const typename Mesh::Cell &tetrahedron = mesh_pointer->template getEntity<Mesh::getMeshDimension()>(
                    tetrahedron_global_index);

            Intersection result;
            result.distance = std::numeric_limits<Real>::max();
            result.nearest_face_global_index = -1;

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
                    if (current_distance < result.distance) {
                        second_minimal_distance = result.distance;
                        result.distance = current_distance;
                        result.nearest_face_global_index = face_global_index;
                    } else if (current_distance < second_minimal_distance) {
                        second_minimal_distance = current_distance;
                    }
                }
            }

            if (second_minimal_distance - result.distance < epsilon) {
                result.is_intersection_with_triangle = false;
            }

            return result;
        }

        /// Get tetrahedron with point
        /// \param mesh_pointer Pointer to device mesh
        /// \param mesh_size Number of tetrahedrons in mesh
        /// \param point Point for test
        /// \return Tetrahedron global index - if point is inside mesh, {} - if otherwise
        static std::optional<Index> get_current_tetrahedron(const Mesh *mesh_pointer, Index mesh_size, const Point &point)
        {
            Array<std::optional<Index>, DeviceType> device_array(mesh_size);
            auto view = device_array.getView();

            Algorithms::ParallelFor<DeviceType>::exec(Index(), mesh_size, [=] __cuda_callable__ (Index i) mutable {
                std::optional<Index> result{};
                if (check_point_in_tetrahedron(mesh_pointer, i, point)) {
                    result = i;
                }
                view[i] = result;
            });

            auto fetch = [=] __cuda_callable__ (Index i) {
                return view[i];
            };

            auto reduction = [=] __cuda_callable__ (std::optional<Index> x, std::optional<Index> y) {
                return x ? x : y;
            };

            return Algorithms::reduce<DeviceType>(Index(), mesh_size, fetch, reduction, std::optional<Index>{});
        }
    };
}
