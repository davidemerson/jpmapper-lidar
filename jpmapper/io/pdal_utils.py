"""
PDAL pipeline utilities for raster operations.
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Define default global flags
PDAL_AVAILABLE = False
Pipeline = None

# Try to import PDAL, but don't fail if not available
try:
    import pdal
    from pdal import Pipeline
    PDAL_AVAILABLE = True
except ImportError:
    # Define a placeholder for type hints
    class PipelinePlaceholder:
        """Placeholder for PDAL Pipeline class when pdal is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PDAL is not installed")
            
    # Assign the placeholder if real class not available
    if Pipeline is None:
        Pipeline = PipelinePlaceholder

logger = logging.getLogger(__name__)


def run_pdal_pipeline(
    pipeline_json: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run a PDAL pipeline with the given JSON configuration.
    
    Args:
        pipeline_json: PDAL pipeline configuration as a dictionary
        output_path: Optional path to save pipeline output
        
    Returns:
        Dictionary with pipeline execution results
        
    Raises:
        RuntimeError: If PDAL execution fails
    """
    try:
        # Try to use python-pdal if available
        if PDAL_AVAILABLE:
            # Update output path if provided
            if output_path:
                for stage in pipeline_json.get("pipeline", []):
                    if isinstance(stage, dict) and "filename" in stage and stage.get("type") in ["writers.gdal", "writers.las"]:
                        stage["filename"] = str(output_path)
            
            # Execute pipeline
            pipeline = Pipeline(json.dumps(pipeline_json))
            count = pipeline.execute()
            
            # Get metadata if available
            metadata = {}
            try:
                metadata = json.loads(pipeline.metadata)
            except Exception:
                pass
                
            result = {
                "success": True,
                "count": count,
                "output": output_path,
                "metadata": metadata
            }
            return result
            
        else:
            logger.warning("python-pdal not available, falling back to CLI")
            # Fall back to CLI if python-pdal not available
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                json_path = tmp.name
                tmp.write(json.dumps(pipeline_json))
            
            try:
                # Run PDAL pipeline
                proc = subprocess.run(
                    ["pdal", "pipeline", json_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Parse metadata if available
                metadata = {}
                try:
                    metadata_path = f"{json_path}.log"
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                except Exception:
                    pass
                
                result = {
                    "success": True,
                    "output": output_path,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "metadata": metadata
                }
                
                return result
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(json_path)
                    metadata_path = f"{json_path}.log"
                    if os.path.exists(metadata_path):
                        os.unlink(metadata_path)
                except Exception:
                    pass
                    
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"PDAL pipeline execution failed: {e.stderr}") from e
    except Exception as e:
        raise RuntimeError(f"Error running PDAL pipeline: {e}") from e
