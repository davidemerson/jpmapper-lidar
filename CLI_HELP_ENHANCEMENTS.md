# CLI Help Documentation Enhancements

## Overview

The JPMapper CLI help system has been significantly enhanced to provide comprehensive onscreen documentation with usage examples, detailed parameter descriptions, and workflow guidance.

## Enhanced Features

### Main CLI (jpmapper --help)
- **Overview**: Enhanced description highlighting advanced LiDAR processing capabilities
- **Quick Start Examples**: Added common usage patterns for immediate reference
- **Feature Highlights**: Mentions metadata-aware rasterization and shapefile filtering
- **Navigation Guidance**: Clear instructions for accessing detailed help

### Filter Commands (jpmapper filter)

#### Main Filter Help (jpmapper filter --help)
- **Comprehensive Overview**: Describes both bounding box and shapefile filtering
- **Usage Examples**: Practical examples for both filtering methods
- **Dependency Information**: Clear notes about enhanced shapefile requirements

#### Bounding Box Filtering (jpmapper filter bbox --help)
- **Detailed Description**: Explains coordinate system requirements and format
- **Multiple Examples**: Shows directory, single file, and coordinate examples
- **Parameter Clarification**: Enhanced descriptions for src, bbox, and dst parameters

#### Shapefile Filtering (jpmapper filter shapefile --help)
- **Enhanced Description**: Explains geometry support and CRS validation
- **Practical Examples**: Shows various usage patterns with different options
- **Parameter Details**: Comprehensive descriptions for all options including buffer and CRS validation

### Rasterization Commands (jpmapper rasterize)

#### Main Rasterize Help (jpmapper rasterize --help)
- **Feature Highlights**: Mentions metadata-aware processing and automatic CRS detection
- **Workflow Examples**: Shows common rasterization patterns
- **Performance Guidance**: Tips for optimization and dependency requirements

#### Tile Rasterization (jpmapper rasterize tile --help)
- **Process Explanation**: Clear description of DSM creation from point clouds
- **Usage Examples**: Multiple examples with different parameter combinations
- **Performance Tips**: Guidance on resolution selection based on data characteristics
- **Parameter Details**: Enhanced descriptions for resolution, EPSG, and worker parameters

### Analysis Commands (jpmapper analyze)

#### Main Analyze Help (jpmapper analyze --help)
- **Application Context**: Explains RF planning and line-of-sight analysis use cases
- **Data Requirements**: Clear specification of CSV format requirements
- **Workflow Examples**: Shows progressive complexity from basic to advanced usage
- **Quality Guidelines**: Advice on LAS data coverage and point density

#### CSV Analysis (jpmapper analyze csv --help)
- **Comprehensive Description**: Detailed explanation of processing and output options
- **Format Specification**: Clear requirements for CSV columns (required vs optional)
- **Output Options**: Examples for console, JSON, and interactive map outputs
- **Performance Guidance**: Tips for caching and repeated analysis optimization
- **Parameter Details**: Enhanced descriptions for all analysis parameters

## Key Improvements

1. **Usage Examples**: Every command now includes multiple practical examples
2. **Parameter Clarity**: Enhanced descriptions explain the purpose and format of each parameter
3. **Workflow Guidance**: Help text guides users through common usage patterns
4. **Dependency Information**: Clear notes about enhanced features and requirements
5. **Performance Tips**: Practical advice for optimization and best practices
6. **Error Prevention**: Better descriptions help users avoid common mistakes

## Enhanced Parameter Descriptions

### Common Improvements
- **File Path Parameters**: Clarified single file vs directory behavior
- **Optional Parameters**: Clear indication of when parameters can be omitted
- **Coordinate Systems**: Better explanation of CRS detection and specification
- **Output Options**: Enhanced descriptions of destination and output format options
- **Performance Parameters**: Improved guidance on worker and resolution settings

### Specific Enhancements
- **Bounding Box Format**: Clear specification of coordinate order and formatting
- **Shapefile Support**: Detailed explanation of geometry types and CRS validation
- **Resolution Guidance**: Practical advice on cell size selection
- **CSV Format**: Comprehensive specification of required and optional columns
- **Output Formats**: Clear explanation of console, JSON, and HTML output options

## User Experience Benefits

1. **Self-Documenting**: Users can discover functionality without external documentation
2. **Example-Driven**: Real-world examples make it easy to get started
3. **Progressive Complexity**: Examples range from simple to advanced usage
4. **Error Reduction**: Clear parameter descriptions prevent common mistakes
5. **Workflow Guidance**: Help text suggests appropriate command sequences
6. **Feature Discovery**: Enhanced descriptions highlight advanced capabilities

## Command Summary

```bash
# Main CLI with enhanced overview and quick start examples
jpmapper --help

# Filter commands with comprehensive spatial filtering guidance
jpmapper filter --help
jpmapper filter bbox --help      # Enhanced bbox filtering with examples
jpmapper filter shapefile --help # Advanced shapefile filtering documentation

# Rasterization commands with metadata-aware processing guidance
jpmapper rasterize --help
jpmapper rasterize tile --help   # Comprehensive DSM creation documentation

# Analysis commands with RF planning workflow guidance
jpmapper analyze --help
jpmapper analyze csv --help      # Detailed line-of-sight analysis documentation
```

## Integration with Enhanced Features

The CLI help system now properly documents and promotes the enhanced features:

- **Metadata-Aware Rasterization**: Highlighted in rasterize command help
- **Shapefile Filtering**: Comprehensive documentation with dependency requirements
- **CRS Validation**: Explained in shapefile filtering help
- **Enhanced API**: Referenced throughout with practical examples
- **Performance Optimization**: Guidance provided for resolution and worker settings

This enhancement ensures that users can effectively discover and utilize all the advanced features of the JPMapper toolkit directly from the command line interface.
