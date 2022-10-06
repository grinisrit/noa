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
 * Implemented by: Gregory Dushkin
 */

#pragma once

// STL headers
#include <cassert>
#include <variant>

// TNL headers
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>

namespace noa::utils::domain {

// The following macro switches on variant stored data type and performs a templated action
// WARNING: This is only designed to be used inside of the `Layer` struct
#define LVFOR(VARIANT, WHAT) /* WHAT(auto& v) accepts a Vector<DataType> reference v */\
                switch (VARIANT.index()) {\
                        case 0: /* int8_t */\
                                WHAT(std::get<Vector<std::int8_t>>(VARIANT));\
                                break;\
                        case 1: /* uint8_t */\
                                WHAT(std::get<Vector<std::uint8_t>>(VARIANT));\
                                break;\
                        case 2: /* int16_t */\
                                WHAT(std::get<Vector<std::int16_t>>(VARIANT));\
                                break;\
                        case 3: /* uint16_t */\
                                WHAT(std::get<Vector<std::uint16_t>>(VARIANT));\
                                break;\
                        case 4: /* int32_t */\
                                WHAT(std::get<Vector<std::int32_t>>(VARIANT));\
                                break;\
                        case 5: /* uint32_t */\
                                WHAT(std::get<Vector<std::uint32_t>>(VARIANT));\
                                break;\
                        case 6: /* int64_t */\
                                WHAT(std::get<Vector<std::int64_t>>(VARIANT));\
                                break;\
                        case 7: /* uint64_t */\
                                WHAT(std::get<Vector<std::uint64_t>>(VARIANT));\
                                break;\
                        case 8: /* float */\
                                WHAT(std::get<Vector<float>>(VARIANT));\
                                break;\
                        case 9: /* double */\
                                WHAT(std::get<Vector<double>>(VARIANT));\
                                break;\
                }
// <-- end #define LVFOR(VARIANT, WHAT)

// struct Layer
// A wrapper class to contain one mesh baked data layer
template <typename Device = TNL::Devices::Host, typename Index = std::size_t>
struct Layer {
        /* ----- PUBLIC TYPE ALIASES ----- */
        // Types should be declared inside the `Layer` struct because they depend on layer
        // template parameters
        template <typename DataType> using Vector = TNL::Containers::Vector<DataType, Device, Index>;

        // TNL::Meshes::Readers::MeshReader stores data baked in mesh in an
        // std::vector. We want to use TNL Vectors to possibly store layers
        // on different devices, so we also need to deine our own std::variant
        // for TNL Vectors instead of using TNL::Meshes::Readers::MeshReader::VariantVector
        using VariantVector = std::variant < Vector< std::int8_t >,
                                                Vector< std::uint8_t >,
                                                Vector< std::int16_t >,
                                                Vector< std::uint16_t >,
                                                Vector< std::int32_t >,
                                                Vector< std::uint32_t >,
                                                Vector< std::int64_t >,
                                                Vector< std::uint64_t >,
                                                Vector< float >,
                                                Vector< double > >;

        protected:
        /* ----- PROTECTED DATA MEMBERS ----- */
        VariantVector data;     // Layer data
        Index size;             // Layer size

        public:
        /* ----- PUBLIC DATA MEMBERS ----- */
        std::string     alias = "";             // Alternative layer name
        bool            exportHint = false;     // Should this Layer be saved (only a hint, all Layers could still be saved)
        /* ----- PUBLIC CONSTRUCTOR ----- */
        template <typename DataType>
        Layer(const Index& newSize, const DataType& value = DataType()) {
                init(newSize, value);
        }
        /* ----- PUBLIC METHODS ----- */
        // Initialize layer data and fill with values
        template <typename DataType>
        void init(const Index& newSize, const DataType& value = DataType()) {
                size = newSize;
                data = Vector<DataType>(size, value);
        }

        // Resize layer
        void setSize(const Index& newSize) {
                size = newSize;
                const auto setSize_f = [&] (auto& v) {
                        v.setSize(size);
                };
                LVFOR(data, setSize_f);
        }

        // Get layer size
        const Index& getSize() const { return size; }

        // Set layer data from std::vector of same size
        template <typename DataType>
        void setFrom(const std::vector<DataType>& from) {
                assert(("Layer::setFrom(): Source size must be equal to layer size", size == from.size()));

                data = Vector<DataType>(size);
                std::get<Vector<DataType>>(data) = from;
        }

        // Get layer element
        template <typename DataType>
        DataType& operator[](const Index& index) {
                return std::get<Vector<DataType>>(data)[index];
        }

        template <typename DataType>
        const DataType& operator[](const Index& index) const {
                return std::get<Vector<DataType>>(data)[index];
        }

        // Get layer data vector
        template <typename DataType>
        Vector<DataType>& get() {
                return std::get<Vector<DataType>>(data);
        }

        template <typename DataType>
        const Vector<DataType>& get() const {
                return std::get<Vector<DataType>>(data);
        }

        // Write layer using a TNL mesh writer
        template <typename Writer>
        void writeCellData(Writer& writer, const std::string& fallback_name) const {
                const auto writeCellData_f = [&] (auto& v) {
                        writer.writeCellData(v, (alias.empty()) ? fallback_name : alias);
                };
                LVFOR(data, writeCellData_f);
        }

        template <typename Writer>
        void writeDataArray(Writer& writer, const std::string& fallback_name) const {
                const auto writeDataArray_f = [&] (auto& v) {
                        writer.writeDataArray(v, (alias.empty()) ? fallback_name : alias);
                };
                LVFOR(data, writeDataArray_f);
        }

        template <typename Writer>
        void writePointData(Writer& writer, const std::string& fallback_name) const {
                const auto writePointData_f = [&] (auto& v) {
                        writer.writePointData(v, (alias.empty()) ? fallback_name : alias);
                };
                LVFOR(data, writePointData_f);
        }
}; // <-- struct Layer

// struct LayerManager
// A class to hold a multitude of `Layer`s of the same size
// (you're supposed to have one LayerManager per mesh dimension that you
// want to store data over)
template <typename Device = TNL::Devices::Host, typename Index = int>
struct LayerManager {
        /* ----- PUBLIC TYPE ALIASES ----- */
        using LayerType = Layer<Device, Index>;
        template <typename DataType> using Vector = typename LayerType::template Vector<DataType>;

        protected:
        /* ----- PROTECTED DATA MEMBERS ----- */
        Index size;
        std::vector<Layer<Device, Index>> layers;

        public:
        /* ----- PUBLIC METHODS ----- */
        // Sets size for all stored layers
        void setSize(const Index& newSize) {
                size = newSize;
                for (auto& layer : layers) layer.setSize(size);
        }
        // Return the number of stored layers
        std::size_t count() const { return layers.size(); }

        // Removes all layers
        void clear() {
                while (!layers.empty()) layers.erase(layers.begin());
        }

        // Adds a layer storing a certain data type and returns its index
        template <typename DataType>
        std::size_t add(const DataType& value = DataType()) {
                layers.push_back(LayerType(size, value));
                return layers.size() - 1;
        }

        // Returns selected layer's data
        template <typename DataType>
        Vector<DataType>& get(const std::size_t& index) {
                return layers.at(index).template get<DataType>();
        }

        template <typename DataType>
        const Vector<DataType>& get(const std::size_t& index) const {
                return layers.at(index).template get<DataType>();
        }

        // Returns selected layer (`Layer` object instead of data)
        LayerType& getLayer(const std::size_t& index) {
                return layers.at(index);
        }

        const LayerType& getLayer(const std::size_t& index) const {
                return layers.at(index);
        }
}; // <-- struct LayerManager

} // <-- namespace noa::utils::domain
