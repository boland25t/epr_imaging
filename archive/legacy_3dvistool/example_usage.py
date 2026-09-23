"""
Example usage of the Point Cloud Pipeline

This script demonstrates how to:
1. Load UTM XYZ coordinate data from CSV
2. Create a 3D grid of cell centroids
3. Optionally add scalar data from the original points
4. Export to PLY format for visualization
"""

from point_cloud_pipeline import PointCloudPipeline


def main():
    """
    Example workflow.
    
    Customize the parameters below for your data.
    """
    
    # ===== CONFIGURATION =====
    INPUT_CSV = "your_data.csv"  # Path to your input CSV file
    
    # Column names in your CSV
    X_COLUMN = "easting"         # UTM Easting (or X)
    Y_COLUMN = "northing"        # UTM Northing (or Y)
    Z_COLUMN = "depth"           # Depth or elevation (Z)
    SCALAR_COLUMN = "temperature"  # Column to use for coloring
    
    # Grid parameters
    CELL_SIZE = 10.0             # Size of cube cells in your data units
    BOUNDARY_EXTENSION = 50.0    # How far beyond min/max coords to extend (0 for no extension)
    
    # Output files
    OUTPUT_CSV = "point_cloud_grid.csv"
    OUTPUT_PLY = "point_cloud_grid.ply"
    
    # ===== WORKFLOW =====
    
    # Initialize pipeline
    pipeline = PointCloudPipeline(
        cell_size=CELL_SIZE,
        boundary_extension=BOUNDARY_EXTENSION
    )
    
    # Load your data
    print("Loading data...")
    pipeline.load_csv(
        filepath=INPUT_CSV,
        x_col=X_COLUMN,
        y_col=Y_COLUMN,
        z_col=Z_COLUMN,
        scalar_col=SCALAR_COLUMN
    )
    
    # Create 3D grid
    print("\nCreating 3D grid...")
    grid_df = pipeline.create_3d_grid()
    print(grid_df.head())
    
    # Add scalar values from original data points to nearest grid cells
    # (Optional - comment out if you don't want to aggregate original data)
    print("\nAggregating scalar data to grid cells...")
    pipeline.add_scalar_data(
        filepath=INPUT_CSV,
        x_col=X_COLUMN,
        y_col=Y_COLUMN,
        z_col=Z_COLUMN,
        scalar_col=SCALAR_COLUMN,
        aggregation='mean'  # Use 'mean', 'min', 'max', or 'count'
    )
    
    # Export to CSV
    print("\nExporting to CSV...")
    pipeline.to_csv(OUTPUT_CSV)
    
    # Export to PLY
    print("\nExporting to PLY...")
    pipeline.to_ply(
        filepath=OUTPUT_PLY,
        scalar_col=pipeline.scalar_name,  # Color by the scalar data
        binary=True  # Use binary format for smaller file size
    )
    
    # Print summary
    print("\n" + pipeline.summary())
    
    print("\n✓ Pipeline complete!")
    print(f"  CSV: {OUTPUT_CSV}")
    print(f"  PLY: {OUTPUT_PLY}")


if __name__ == "__main__":
    main()
