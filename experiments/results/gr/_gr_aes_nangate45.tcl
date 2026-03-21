read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef"
read_lef "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef"
read_def "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/OpenROAD/src/drt/test/aes_nangate45_preroute.def"
set_global_routing_layer_adjustment metal1-metal10 0.5
global_route -verbose

puts "=== GCELL_CONGESTION_START ==="
set block [ord::get_db_block]
set die [$block getDieArea]
puts "DIE [$die xMin] [$die yMin] [$die xMax] [$die yMax]"
set dbu [$block getDefUnits]
puts "DBU $dbu"

# Get GCell grid info
set gcell_grid [$block getGCellGrid]
set x_grids [$gcell_grid getXGrids]
set y_grids [$gcell_grid getYGrids]
puts "GCELL_NX [llength $x_grids]"
puts "GCELL_NY [llength $y_grids]"

# Get nets and their routing info
set nets [$block getNets]
puts "NETS [llength $nets]"

# Report congestion per GCell using routing tracks
# We'll use the GRT capacity/usage data from the log
puts "=== GCELL_CONGESTION_END ==="
