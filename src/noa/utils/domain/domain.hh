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
 * \file domain.hh
 * \brief \ref Domain wraps TNL Mesh and provedes a simple interface for storing data over it
 *
 * Implemented by: Gregory Dushkin
 */

#pragma once

// STL headers
#include <optional>

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DefaultConfig.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshBuilder.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/VTUWriter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityCenter.h>

// NOA headers
#include <noa/utils/common.hh>

// Local headers
#include "configtagpermissive.hh"
#include "layermanager.hh"

// Macros definition
/// \ref Domain template arguments list
#define __domain_targs__ \
        typename CellTopology, typename Device, typename Real, typename GlobalIndex, typename LocalIndex
/// Full \ref Domain type with template arguments. To be used with \ref __domain_targs__
#define __DomainType__  \
        noa::utils::domain::Domain<CellTopology, Device, Real, GlobalIndex, LocalIndex>
/// Same as \ref __domain_targs__, but not including CellTopology
#define __domain_targs_topospec__ \
   typename Device, typename Real, typename GlobalIndex, typename LocalIndex
/// Same as \ref __DomainType__, but accepts cell topology as an argument
/// (for use with \ref __domain_targs__topospec__)
#define __DomainTypeTopoSpec__(topology) \
   noa::utils::domain::Domain<topology, Device, Real, GlobalIndex, LocalIndex>

/// Namespace containing the domain-related code
namespace noa::utils::domain {

/// \brief Domain stores a TNL mesh and various data over its elements in one place
/// providing a friendly all-in-one-place interface for solvers.
template <typename CellTopology, typename Device = TNL::Devices::Host, typename Real = float, typename GlobalIndex = long int, typename LocalIndex = short int>
struct Domain {
        /* ----- PUBLIC TYPE ALIASES ----- */
        /// TNL Mesh config
        using MeshConfig        = TNL::Meshes::DefaultConfig<CellTopology, CellTopology::dimension, Real, GlobalIndex, LocalIndex>;
        /// \brief TNL Mesh type
        ///
        /// Doesn't use the \p Device template parameter.
        /// (TODO?) Look into CUDA Mesh implementation in TNL
        using MeshType          = TNL::Meshes::Mesh<MeshConfig>; // Meshes only seem to be implemented on Host (?)
        /// TNL Mesh writer type
        using MeshWriter        = TNL::Meshes::Writers::VTUWriter<MeshType>;
        /// LayerManager type
        using LayerManagerType  = LayerManager<Device, GlobalIndex>;

        /// Floating point numeric type
        using RealType          = Real;
        /// TNL mesh device type
        using DeviceType        = Device;
        /// Mesh global index type
        using GlobalIndexType   = GlobalIndex;
        /// Mesh local index type
        using LocalIndexType    = LocalIndex;

        /// Mesh point type
        using PointType         = typename MeshType::PointType;

        /// Get mesh dimensions
        static constexpr int getMeshDimension() { return MeshType::getMeshDimension(); }

        protected:
        /* ----- PROTECTED DATA MEMBERS ----- */
        /// Domain mesh
        std::optional<MeshType> mesh = std::nullopt;
        /// Mesh data layers
        std::vector<LayerManagerType> layers;

        /* ----- PROTECTED METHODS ----- */
        /// Updates all layer sizes from mesh entities count
        template <int fromDimension = getMeshDimension()>
        void updateLayerSizes() {
                const auto size = isClean() ? 0 : mesh.value().template getEntitiesCount<fromDimension>();
                layers.at(fromDimension).setSize(size);
                if constexpr (fromDimension > 0) updateLayerSizes<fromDimension - 1>();
        }

        public:
        /* ----- PUBLIC METHODS ----- */
        /// Constructor
        Domain() {
                // If default-constructed, generate layers for each of the meshes dimensions
                layers = std::vector<LayerManagerType>(getMeshDimension() + 1);
        }

        /// \brief Clear mesh data
        ///
        /// Resets both mesh and layer data
        void clear() {
                if (isClean()) return; // Nothing to clear

                // Reset the mesh
                mesh.reset();

                // Remove all the layers
                clearLayers();
        }

        /// Clear layer data
        void clearLayers() { for (auto& layer : layers) layer.clear(); }

        /// Check if the domain is empty. Equivalent to `!mesh.has_value()`
        bool isClean() const { return !mesh.has_value(); }

        /// Get mesh as a constant reference
        const MeshType& getMesh() const { return mesh.value(); }

        /// Get center coordinates for entity with index
        template <int dimension>
        auto getEntityCenter(GlobalIndex idx) const {
                return TNL::Meshes::getEntityCenter(*this->mesh, this->mesh->template getEntity<dimension>(idx));
        }

        /// Get layers
        LayerManagerType& getLayers(const std::size_t& dimension) {
                return layers.at(dimension);
        }
        const LayerManagerType& getLayers(const std::size_t& dimension) const {
                return layers.at(dimension);
        }

        /// \brief Cell layers requested for loadFrom() function
        ///
        /// Contains pairs of ( layer_name -> layer_key )
        using LayersRequest = std::map<std::string, std::size_t>;

        private:
        /// \brief Layer loading helper
        ///
        /// \tparam DataType - required layer data type
        /// \tparam FromType - TNL's MeshReader::VariantVector, gets deduced automatically to avoid hardcoding
        ///
        /// \param from - VariantVector with data
        ///
        /// Handles loading a layer from a variant of `std::vector`'s when the
        /// layer data type is known
        template <typename DataType, typename FromType>
        void load_layer(std::size_t key, const FromType& from) {
                const auto& fromT = std::get<std::vector<DataType>>(from);
                auto& toT = this->getLayers(this->getMeshDimension()).
                                                template add<DataType>(key).template get<DataType>();

                for (std::size_t i = 0; i < fromT.size(); ++i)
                        toT[i] = fromT[i];
        }

        public:
        /// \brief Load Domain from a file
        /// \param filename - path to mesh file
        /// \param layersRequest - a list of requested cell layers
        ///
        /// Only supports loading of the cell layers (layers for dimension reported by getMeshDimension()).
        /// Layers are loaded in order in which they were specified in \p layersRequest.
        void loadFrom(const Path& filename, const LayersRequest& layersRequest = {}) {
		if (!isClean())
			throw std::runtime_error("Mesh data is not empty, cannot load!");

                if (!std::filesystem::exists(filename))
                        throw std::runtime_error("Mesh file not found: " + filename.string() + "!");

                auto loader = [&] (auto& reader, auto&& loadedMesh) {
                        using LoadedTypeRef = decltype(loadedMesh);
                        using LoadedType = typename std::remove_reference<LoadedTypeRef>::type;

                        if constexpr (std::is_same_v<MeshType, LoadedType>) {
                                mesh = loadedMesh;
                                updateLayerSizes();
                        } else throw std::runtime_error("Read mesh type differs from expected!");

                        // Load cell layers
                        for (auto& [ name, index ] : layersRequest) {
                                const auto data = reader.readCellData(name);

                                switch (data.index()) {
                                        case 0: /* int8_t */
                                                load_layer<std::int8_t>(index, data);
                                                break;
                                        case 1: /* uint8_t */
                                                load_layer<std::uint8_t>(index, data);
                                                break;
                                        case 2: /* int16_t */
                                                load_layer<std::int16_t>(index, data);
                                                break;
                                        case 3: /* uint16_t */
                                                load_layer<std::uint16_t>(index, data);
                                                break;
                                        case 4: /* int32_t */
                                                load_layer<std::int32_t>(index, data);
                                                break;
                                        case 5: /* uint32_t */
                                                load_layer<std::uint32_t>(index, data);
                                                break;
                                        case 6: /* int64_t */
                                                load_layer<std::int64_t>(index, data);
                                                break;
                                        case 7: /* uint64_t */
                                                load_layer<std::uint64_t>(index, data);
                                                break;
                                        case 8: /* float */
                                                load_layer<float>(index, data);
                                                break;
                                        case 9: /* double */
                                                load_layer<double>(index, data);
                                                break;
                                }

                                this->getLayers(this->getMeshDimension()).getLayer(index).alias = name;
                                this->getLayers(this->getMeshDimension()).getLayer(index).exportHint = true;
                        }

                        return true;
                };

                using ConfigTag = ConfigTagPermissive<CellTopology>;
                if (!TNL::Meshes::resolveAndLoadMesh<ConfigTag, TNL::Devices::Host>(loader, filename, "auto"))
                        throw std::runtime_error("Could not load mesh (resolveAndLoadMesh returned `false`)!");
        }

        /// Write Domain to a file
        void write(const Path& filename) {
                assert(("Mesh data is empty, nothing to save!", !isClean()));

                std::ofstream file(filename);
                if (!file.is_open())
                        throw std::runtime_error("Cannot open file " + filename.string() + "!");

                writeStream(file);
        }

        /// Write domain to an ostream
        void writeStream(std::ostream& stream) {
                MeshWriter writer(stream);
                writer.template writeEntities<getMeshDimension()>(mesh.value());

                // Write layers
                for (int dim = 0; dim <= getMeshDimension(); ++dim)
                        for (auto it : layers.at(dim)) {
                                const auto& layer = it.second;
                                if (!layer.exportHint) continue;
                                if (dim == getMeshDimension())
                                        layer.writeCellData(writer, "cell_layer_" + std::to_string(it.first));
                                else if (dim == 0)
                                        layer.writePointData(writer, "point_layer_" + std::to_string(it.first));
                                else
                                        layer.writeDataArray(writer, "dim" + std::to_string(dim) + "_layer_" + std::to_string(it.first));
                        }
        }

        /* ----- DOMAIN MESH GENERATORS ----- */
        // Declare as friends since they're supposed to modify mesh
        template <typename Device2, typename Real2, typename GlobalIndex2, typename LocalIndex2>
        friend void generate2DGrid(Domain<TNL::Meshes::Topologies::Triangle, Device2, Real2, GlobalIndex2, LocalIndex2>& domain,
                                std::size_t Nx,
                                std::size_t Ny,
                                float dx,
                                float dy,
                                float offsetX,
                                float offsetY);
}; // <-- struct Domain

/// \brief Generates a uniform 2D grid mesh for the domain. Implementation depends on \p CellTopology2
template <typename CellTopology2, typename Device2, typename Real2, typename GlobalIndex2, typename LocalIndex2>
void generate2DGrid(Domain<CellTopology2, Device2, Real2, GlobalIndex2, LocalIndex2>& domain,
                        std::size_t Nx,
                        std::size_t Ny,
                        float dx,
                        float dy,
                        float offsetX = 0,
                        float offsetY = 0) {
        throw std::runtime_error("generate2DGrid is not implemented for this topology!");
}

/// \brief Specialization for Triangle topology
template <typename Device2, typename Real2, typename GlobalIndex2, typename LocalIndex2>
void generate2DGrid(Domain<TNL::Meshes::Topologies::Triangle, Device2, Real2, GlobalIndex2, LocalIndex2>& domain,
                        std::size_t Nx,
                        std::size_t Ny,
                        float dx,
                        float dy,
                        float offsetX = 0,
                        float offsetY = 0) {
        assert(("Mesh data is not empty, cannot create grid!", domain.isClean()));

        using CellTopology = TNL::Meshes::Topologies::Triangle;
        using DomainType = Domain<CellTopology, Device2, Real2, GlobalIndex2, LocalIndex2>;
        using MeshType = typename DomainType::MeshType;

        domain.mesh = MeshType();

        using Builder = TNL::Meshes::MeshBuilder<MeshType>;
        Builder builder;

        const auto elems = Nx * Ny * 2;
        const auto points = (Nx + 1) * (Ny + 1);
        builder.setEntitiesCount(points, elems);

        using PointType = typename MeshType::MeshTraitsType::PointType;
        const auto pointId = [&] (const int& ix, const int& iy) -> GlobalIndex2 {
                return ix + (Nx + 1) * iy;
        };
        const auto fillPoints = [&] (const int& ix, const int& iy) {
                builder.setPoint(pointId(ix, iy), PointType(ix * dx + offsetX, iy * dy + offsetY));
        };
        TNL::Algorithms::ParallelFor2D<TNL::Devices::Host>::exec(std::size_t{0}, std::size_t{0}, Nx + 1, Ny + 1, fillPoints);

        const auto fillElems = [&] (const int& ix, const int& iy, const int& u) {
                const auto cell = 2 * (ix + Nx * iy) + u;
                auto seed = builder.getCellSeed(cell);

                switch (u) {
                        case 1:
                                seed.setCornerId(0, pointId(ix, iy));
                                seed.setCornerId(1, pointId(ix, iy + 1));
                                seed.setCornerId(2, pointId(ix + 1, iy));
                                break;
                        case 0:
                                seed.setCornerId(0, pointId(ix + 1, iy + 1));
                                seed.setCornerId(1, pointId(ix + 1, iy));
                                seed.setCornerId(2, pointId(ix, iy + 1));
                                break;
                }
        };
        TNL::Algorithms::ParallelFor3D<TNL::Devices::Host>::exec(std::size_t{0}, std::size_t{0}, std::size_t{0}, Nx, Ny, std::size_t{2}, fillElems);

        builder.build(domain.mesh.value());

        domain.updateLayerSizes();
}

} // <-- namespace noa::utils::domain
