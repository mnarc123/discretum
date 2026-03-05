#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <set>
#include <regex>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <fmt/format.h>
#include "core/graph.hpp"
#include "utils/logging.hpp"

namespace discretum {
namespace io {

/**
 * @brief Binary format version for compatibility checking
 */
constexpr uint32_t BINARY_FORMAT_VERSION = 1;

/**
 * @brief Magic number for file format identification
 */
constexpr uint32_t MAGIC_NUMBER = 0x44495343; // "DISC"

/**
 * @brief Compute CRC32 checksum over a byte buffer
 * @param data Pointer to data
 * @param length Number of bytes
 * @return CRC32 checksum
 */
inline uint32_t crc32(const void* data, size_t length) {
    const uint8_t* buf = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc ^= buf[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

/**
 * @brief File header for binary graph format
 */
struct GraphFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_nodes;
    uint32_t num_edges;
    uint64_t timestamp;
    uint32_t checksum;
    uint32_t reserved[10];
};

/**
 * @brief Save graph to binary file with metadata
 * @param graph Graph to save
 * @param path Output file path
 * @param compress Whether to compress the data
 */
inline void save_graph_binary(const DynamicGraph& graph, const std::string& path, bool compress = false) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    
    // Prepare header
    GraphFileHeader header{};
    header.magic = MAGIC_NUMBER;
    header.version = BINARY_FORMAT_VERSION;
    header.num_nodes = graph.num_nodes();
    header.num_edges = graph.num_edges();
    header.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    // Compute checksum over node and adjacency data
    const auto& nodes = graph.get_nodes();
    const auto& adjacency = graph.get_adjacency();
    uint32_t adj_size = static_cast<uint32_t>(adjacency.size());
    {
        // Build contiguous buffer for checksum: nodes + adj_size + adjacency
        std::vector<uint8_t> buf;
        buf.reserve(nodes.size() * sizeof(Node) + sizeof(adj_size) + adj_size * sizeof(uint32_t));
        auto append = [&](const void* ptr, size_t len) {
            const uint8_t* p = static_cast<const uint8_t*>(ptr);
            buf.insert(buf.end(), p, p + len);
        };
        for (const auto& node : nodes)
            append(&node, sizeof(node));
        append(&adj_size, sizeof(adj_size));
        if (adj_size > 0)
            append(adjacency.data(), adj_size * sizeof(uint32_t));
        header.checksum = crc32(buf.data(), buf.size());
    }
    
    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Write nodes
    for (const auto& node : nodes) {
        file.write(reinterpret_cast<const char*>(&node), sizeof(node));
    }
    
    // Write adjacency
    file.write(reinterpret_cast<const char*>(&adj_size), sizeof(adj_size));
    file.write(reinterpret_cast<const char*>(adjacency.data()), adj_size * sizeof(uint32_t));
    
    file.close();
    
    // Log file size
    auto file_size = std::filesystem::file_size(path);
    spdlog::info("Graph saved to '{}' ({:.2f} MB)", path, file_size / (1024.0 * 1024.0));
}

/**
 * @brief Load graph from binary file
 * @param path Input file path
 * @return Loaded graph
 */
inline DynamicGraph load_graph_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    
    // Read and validate header
    GraphFileHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.magic != MAGIC_NUMBER) {
        throw std::runtime_error("Invalid file format: wrong magic number");
    }
    
    if (header.version != BINARY_FORMAT_VERSION) {
        throw std::runtime_error(fmt::format("Unsupported format version: {} (expected {})", 
                                           header.version, BINARY_FORMAT_VERSION));
    }
    
    spdlog::info("Loading graph: {} nodes, {} edges", header.num_nodes, header.num_edges);
    
    // Read payload and verify checksum
    if (header.checksum != 0) {
        // Read all remaining bytes for checksum verification
        auto payload_start = file.tellg();
        file.seekg(0, std::ios::end);
        auto payload_size = file.tellg() - payload_start;
        file.seekg(payload_start);
        std::vector<uint8_t> payload(payload_size);
        file.read(reinterpret_cast<char*>(payload.data()), payload_size);
        
        uint32_t computed = crc32(payload.data(), payload.size());
        if (computed != header.checksum) {
            throw std::runtime_error(fmt::format(
                "Checksum mismatch: expected 0x{:08X}, got 0x{:08X}",
                header.checksum, computed));
        }
        spdlog::debug("Checksum verified: 0x{:08X}", computed);
    }
    
    // Use the existing load method from DynamicGraph
    file.close();
    return DynamicGraph::load(path);
}

/**
 * @brief Save graph to human-readable text format
 * @param graph Graph to save
 * @param path Output file path
 */
inline void save_graph_text(const DynamicGraph& graph, const std::string& path) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    
    // Write header
    file << "# DISCRETUM Graph Format v1.0\n";
    file << "# Nodes: " << graph.num_nodes() << "\n";
    file << "# Edges: " << graph.num_edges() << "\n";
    file << "\n";
    
    // Write nodes section
    file << "# Node format: node_id state degree\n";
    file << "NODES\n";
    
    const auto& nodes = graph.get_nodes();
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        if (graph.degree(i) > 0 || nodes[i].state > 0) {
            file << i << " " << static_cast<int>(nodes[i].state) << " " << graph.degree(i) << "\n";
        }
    }
    
    // Write edges section
    file << "\n# Edge format: source target\n";
    file << "EDGES\n";
    
    std::set<std::pair<uint32_t, uint32_t>> written_edges;
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        for (uint32_t neighbor : graph.neighbors(i)) {
            auto edge = std::minmax(i, neighbor);
            if (written_edges.find(edge) == written_edges.end()) {
                file << edge.first << " " << edge.second << "\n";
                written_edges.insert(edge);
            }
        }
    }
    
    file.close();
    spdlog::info("Graph saved to text format: {}", path);
}

/**
 * @brief Save evolution state for checkpointing
 * @param graph Current graph
 * @param step Current evolution step
 * @param params Evolution parameters
 * @param path Output file path
 */
inline void save_checkpoint(const DynamicGraph& graph, uint32_t step, 
                          const std::vector<double>& params, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open checkpoint file for writing: " + path);
    }
    
    // Write checkpoint header
    uint32_t magic = 0x434B5054; // "CKPT"
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&step), sizeof(step));
    
    // Write parameters
    uint32_t num_params = params.size();
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
    file.write(reinterpret_cast<const char*>(params.data()), num_params * sizeof(double));
    
    // Write timestamp
    uint64_t timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    file.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    
    file.close();
    
    // Save graph separately
    std::string graph_path = path + ".graph";
    save_graph_binary(graph, graph_path);
    
    spdlog::info("Checkpoint saved at step {}: {}", step, path);
}

/**
 * @brief Load checkpoint
 * @param path Checkpoint file path
 * @param graph Output graph
 * @param step Output step
 * @param params Output parameters
 */
inline void load_checkpoint(const std::string& path, DynamicGraph& graph, 
                          uint32_t& step, std::vector<double>& params) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open checkpoint file for reading: " + path);
    }
    
    // Read and validate header
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x434B5054) {
        throw std::runtime_error("Invalid checkpoint format");
    }
    
    file.read(reinterpret_cast<char*>(&step), sizeof(step));
    
    // Read parameters
    uint32_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    params.resize(num_params);
    file.read(reinterpret_cast<char*>(params.data()), num_params * sizeof(double));
    
    // Read timestamp
    uint64_t timestamp;
    file.read(reinterpret_cast<char*>(&timestamp), sizeof(timestamp));
    
    file.close();
    
    // Load graph
    std::string graph_path = path + ".graph";
    graph = load_graph_binary(graph_path);
    
    // Convert timestamp to readable format
    auto time_point = std::chrono::system_clock::time_point(std::chrono::nanoseconds(timestamp));
    auto time_t = std::chrono::system_clock::to_time_t(time_point);
    
    spdlog::info("Checkpoint loaded from step {} (saved at {})", step, std::ctime(&time_t));
}

/**
 * @brief Create directory if it doesn't exist
 * @param path Directory path
 */
inline void ensure_directory(const std::string& path) {
    std::filesystem::create_directories(path);
}

/**
 * @brief Generate timestamped filename
 * @param prefix Filename prefix
 * @param suffix Filename suffix/extension
 * @return Timestamped filename
 */
inline std::string timestamped_filename(const std::string& prefix, const std::string& suffix) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << prefix << "_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << suffix;
    return ss.str();
}

/**
 * @brief List files matching pattern in directory
 * @param directory Directory path
 * @param pattern Filename pattern (e.g., "checkpoint_*.bin")
 * @return Vector of matching file paths
 */
inline std::vector<std::string> list_files(const std::string& directory, const std::string& pattern) {
    std::vector<std::string> files;
    
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // Convert glob pattern to regex: escape specials, * → .*, ? → .
            std::string regex_str;
            for (char c : pattern) {
                switch (c) {
                    case '*': regex_str += ".*"; break;
                    case '?': regex_str += "."; break;
                    case '.': case '(': case ')': case '[': case ']':
                    case '{': case '}': case '+': case '^': case '$':
                    case '|': case '\\': regex_str += '\\'; regex_str += c; break;
                    default: regex_str += c; break;
                }
            }
            std::regex re(regex_str);
            if (std::regex_match(filename, re)) {
                files.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace io
} // namespace discretum
