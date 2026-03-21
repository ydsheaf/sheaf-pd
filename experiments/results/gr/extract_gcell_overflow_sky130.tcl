# extract_gcell_overflow_sky130.tcl
# Extract per-GCell congestion data after congested global routing on SKY130
# Independent ground truth for routing congestion (Issue #19)
#
# SKY130 layers: li1, met1, met2, met3, met4, met5
# Uses congestion-inducing settings: met1-met5 0.95 adjustment, met2-met3 signal

set ORFS "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"
read_lef "$ORFS/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef"
read_lef "$ORFS/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef"
read_def "$ORFS/flow/tutorials/scripts/drt/gcd/4_cts.def"
set OUT_DIR "/mnt/storage1/users/ydwu/claude_projects/sheaf-pd/experiments/results/gr"

# --- Get layer info ---
set db [ord::get_db]
set tech [$db getTech]
set block [ord::get_db_block]

array set layer_pitch {}
array set layer_dir {}
for {set i 1} {$i <= 10} {incr i} {
    catch {
        set layer [$tech findRoutingLayer $i]
        if {$layer != "NULL"} {
            set lname [$layer getName]
            set layer_pitch($lname) [$layer getPitch]
            set layer_dir($lname) [$layer getDirection]
        }
    }
}

# Print discovered layers
puts "=== Layer info ==="
foreach lname [lsort [array names layer_pitch]] {
    puts "  $lname: pitch=$layer_pitch($lname) dir=$layer_dir($lname)"
}

# --- Run congested global route (same settings that produced 3990 overflow) ---
set_global_routing_layer_adjustment met1-met5 0.95
set_routing_layers -signal met2-met3
global_route -verbose -allow_congestion

# --- Get GCell grid ---
set gcell_grid [$block getGCellGrid]
set x_grids [$gcell_grid getGridX]
set y_grids [$gcell_grid getGridY]
set nx [expr {[llength $x_grids] - 1}]
set ny [expr {[llength $y_grids] - 1}]
puts "GCell grid: $nx x $ny"

# --- Write segments ---
set seg_file "$OUT_DIR/sky130_congested_segments.txt"
grt::write_segments $seg_file

# --- Parse segments and compute edge-based demand ---
proc find_gcell_idx {coord grid_list} {
    set n [llength $grid_list]
    for {set i 0} {$i < [expr {$n - 1}]} {incr i} {
        if {$coord < [lindex $grid_list [expr {$i + 1}]]} {
            return $i
        }
    }
    return [expr {$n - 2}]
}

# Edge demand arrays
array set h_edge_demand {}
array set v_edge_demand {}

# Parse segments file
set fp [open $seg_file r]
set net_name ""
set in_net 0
set seg_count 0

while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line eq "("} {
        set in_net 1
        continue
    }
    if {$line eq ")"} {
        set in_net 0
        continue
    }
    if {!$in_net} {
        set net_name $line
        continue
    }
    # Parse segment: x1 y1 layer1 x2 y2 layer2
    set parts [split $line " "]
    if {[llength $parts] != 6} continue
    lassign $parts x1 y1 l1 x2 y2 l2

    # Skip vias (same coordinates, different layers)
    if {$x1 == $x2 && $y1 == $y2} continue

    # Wire segment on same layer
    if {$l1 ne $l2} continue

    set lname $l1

    # Determine which gcell boundaries this segment crosses
    if {$y1 == $y2} {
        # Horizontal segment - crosses vertical (X) gcell boundaries
        set xlo [expr {min($x1, $x2)}]
        set xhi [expr {max($x1, $x2)}]
        set gy [find_gcell_idx $y1 $y_grids]

        set gx_lo [find_gcell_idx $xlo $x_grids]
        set gx_hi [find_gcell_idx $xhi $x_grids]

        for {set gx $gx_lo} {$gx < $gx_hi} {incr gx} {
            set key "${gx},${gy},${lname}"
            if {[info exists h_edge_demand($key)]} {
                incr h_edge_demand($key)
            } else {
                set h_edge_demand($key) 1
            }
        }
    } elseif {$x1 == $x2} {
        # Vertical segment - crosses horizontal (Y) gcell boundaries
        set ylo [expr {min($y1, $y2)}]
        set yhi [expr {max($y1, $y2)}]
        set gx [find_gcell_idx $x1 $x_grids]

        set gy_lo [find_gcell_idx $ylo $y_grids]
        set gy_hi [find_gcell_idx $yhi $y_grids]

        for {set gy $gy_lo} {$gy < $gy_hi} {incr gy} {
            set key "${gx},${gy},${lname}"
            if {[info exists v_edge_demand($key)]} {
                incr v_edge_demand($key)
            } else {
                set v_edge_demand($key) 1
            }
        }
    }
    incr seg_count
}
close $fp
puts "Processed $seg_count wire segments"

# --- Also compute guide-based per-GCell demand ---
array set gcell_demand {}
set guide_count 0
set nets [$block getNets]
foreach net $nets {
    set guides [$net getGuides]
    foreach guide $guides {
        set box [$guide getBox]
        set layer [$guide getLayer]
        set lname [$layer getName]
        set xlo [$box xMin]
        set ylo [$box yMin]
        set xhi [$box xMax]
        set yhi [$box yMax]

        set gx_lo [find_gcell_idx $xlo $x_grids]
        set gx_hi [find_gcell_idx $xhi $x_grids]
        set gy_lo [find_gcell_idx $ylo $y_grids]
        set gy_hi [find_gcell_idx $yhi $y_grids]

        for {set gx $gx_lo} {$gx <= $gx_hi} {incr gx} {
            for {set gy $gy_lo} {$gy <= $gy_hi} {incr gy} {
                set key "${gx},${gy},${lname}"
                if {[info exists gcell_demand($key)]} {
                    incr gcell_demand($key)
                } else {
                    set gcell_demand($key) 1
                }
            }
        }
        incr guide_count
    }
}
puts "Processed $guide_count guides"

# --- Compute edge capacities and overflow ---
puts "\n=== Edge-based overflow statistics ==="
set total_h_overflow 0
set total_v_overflow 0
set h_edges_overflow 0
set v_edges_overflow 0
set max_h_overflow 0
set max_v_overflow 0
set total_h_demand 0
set total_v_demand 0
set total_h_capacity 0
set total_v_capacity 0

# Check horizontal edge overflow
foreach key [array names h_edge_demand] {
    set parts [split $key ","]
    set gx [lindex $parts 0]
    set gy [lindex $parts 1]
    set lname [lindex $parts 2]
    set demand $h_edge_demand($key)

    set pitch $layer_pitch($lname)
    set dir $layer_dir($lname)
    set capacity 0
    if {$pitch > 0} {
        if {$dir == "VERTICAL"} {
            set gcell_w [expr {[lindex $x_grids [expr {$gx + 1}]] - [lindex $x_grids $gx]}]
            set capacity [expr {$gcell_w / $pitch}]
        } else {
            set gcell_h [expr {[lindex $y_grids [expr {$gy + 1}]] - [lindex $y_grids $gy]}]
            set capacity [expr {$gcell_h / $pitch}]
        }
    }

    set overflow [expr {max(0, $demand - $capacity)}]
    incr total_h_demand $demand
    incr total_h_capacity $capacity
    incr total_h_overflow $overflow
    if {$overflow > 0} { incr h_edges_overflow }
    if {$overflow > $max_h_overflow} { set max_h_overflow $overflow }
}

# Check vertical edge overflow
foreach key [array names v_edge_demand] {
    set parts [split $key ","]
    set gx [lindex $parts 0]
    set gy [lindex $parts 1]
    set lname [lindex $parts 2]
    set demand $v_edge_demand($key)

    set pitch $layer_pitch($lname)
    set dir $layer_dir($lname)
    set capacity 0
    if {$pitch > 0} {
        if {$dir == "HORIZONTAL"} {
            set gcell_h [expr {[lindex $y_grids [expr {$gy + 1}]] - [lindex $y_grids $gy]}]
            set capacity [expr {$gcell_h / $pitch}]
        } else {
            set gcell_w [expr {[lindex $x_grids [expr {$gx + 1}]] - [lindex $x_grids $gx]}]
            set capacity [expr {$gcell_w / $pitch}]
        }
    }

    set overflow [expr {max(0, $demand - $capacity)}]
    incr total_v_demand $demand
    incr total_v_capacity $capacity
    incr total_v_overflow $overflow
    if {$overflow > 0} { incr v_edges_overflow }
    if {$overflow > $max_v_overflow} { set max_v_overflow $overflow }
}

puts "H-edges: demand=$total_h_demand capacity=$total_h_capacity overflow=$total_h_overflow (${h_edges_overflow} edges)"
puts "V-edges: demand=$total_v_demand capacity=$total_v_capacity overflow=$total_v_overflow (${v_edges_overflow} edges)"
puts "Total overflow: [expr {$total_h_overflow + $total_v_overflow}]"
puts "Max H overflow: $max_h_overflow, Max V overflow: $max_v_overflow"

# --- Build JSON output ---
set json_file "$OUT_DIR/gcell_overflow_congested_gcd_sky130.json"
set fp [open $json_file w]

puts $fp "\{"
puts $fp "  \"design\": \"gcd_sky130\","
puts $fp "  \"grid_nx\": $nx,"
puts $fp "  \"grid_ny\": $ny,"
puts $fp "  \"x_grids\": \[[join $x_grids ", "]\],"
puts $fp "  \"y_grids\": \[[join $y_grids ", "]\],"

# Layer info
puts $fp "  \"layer_info\": \{"
set first_layer 1
foreach lname [lsort [array names layer_pitch]] {
    if {!$first_layer} { puts $fp "," }
    set first_layer 0
    puts -nonewline $fp "    \"$lname\": \{\"direction\": \"$layer_dir($lname)\", \"pitch_dbu\": $layer_pitch($lname)\}"
}
puts $fp ""
puts $fp "  \},"

# GCell data: aggregate across layers for each (gx,gy)
puts $fp "  \"gcells\": \["

array set agg_cap {}
array set agg_use {}

foreach key [array names gcell_demand] {
    set parts [split $key ","]
    set gx [lindex $parts 0]
    set gy [lindex $parts 1]
    set lname [lindex $parts 2]
    set demand $gcell_demand($key)

    set pitch $layer_pitch($lname)
    set dir $layer_dir($lname)
    set capacity 0
    if {$pitch > 0} {
        if {$dir == "HORIZONTAL"} {
            set gcell_h [expr {[lindex $y_grids [expr {$gy + 1}]] - [lindex $y_grids $gy]}]
            set capacity [expr {$gcell_h / $pitch}]
        } else {
            set gcell_w [expr {[lindex $x_grids [expr {$gx + 1}]] - [lindex $x_grids $gx]}]
            set capacity [expr {$gcell_w / $pitch}]
        }
    }

    set akey "${gx},${gy}"
    if {[info exists agg_cap($akey)]} {
        set agg_cap($akey) [expr {$agg_cap($akey) + $capacity}]
        set agg_use($akey) [expr {$agg_use($akey) + $demand}]
    } else {
        set agg_cap($akey) $capacity
        set agg_use($akey) $demand
    }
}

set first 1
set total_of 0
set n_of 0
foreach akey [lsort -dictionary [array names agg_cap]] {
    set parts [split $akey ","]
    set gx [lindex $parts 0]
    set gy [lindex $parts 1]
    set cap $agg_cap($akey)
    set use $agg_use($akey)
    set of [expr {max(0, $use - $cap)}]
    if {$of > 0} { incr n_of; incr total_of $of }
    if {!$first} { puts $fp "," }
    set first 0
    puts -nonewline $fp "    \{\"gx\": $gx, \"gy\": $gy, \"capacity\": $cap, \"usage\": $use, \"overflow\": $of\}"
}

puts $fp ""
puts $fp "  \],"

# Edge-based data
puts $fp "  \"edges\": \["
set first 1

# H-edges
foreach key [lsort -dictionary [array names h_edge_demand]] {
    set parts [split $key ","]
    set gx [lindex $parts 0]
    set gy [lindex $parts 1]
    set lname [lindex $parts 2]
    set demand $h_edge_demand($key)

    set pitch $layer_pitch($lname)
    set dir $layer_dir($lname)
    set capacity 0
    if {$pitch > 0} {
        if {$dir == "VERTICAL"} {
            set gcell_w [expr {[lindex $x_grids [expr {$gx + 1}]] - [lindex $x_grids $gx]}]
            set capacity [expr {$gcell_w / $pitch}]
        } else {
            set gcell_h [expr {[lindex $y_grids [expr {$gy + 1}]] - [lindex $y_grids $gy]}]
            set capacity [expr {$gcell_h / $pitch}]
        }
    }
    set overflow [expr {max(0, $demand - $capacity)}]
    if {!$first} { puts $fp "," }
    set first 0
    puts -nonewline $fp "    \{\"type\": \"H\", \"gx\": $gx, \"gy\": $gy, \"layer\": \"$lname\", \"capacity\": $capacity, \"demand\": $demand, \"overflow\": $overflow\}"
}

# V-edges
foreach key [lsort -dictionary [array names v_edge_demand]] {
    set parts [split $key ","]
    set gx [lindex $parts 0]
    set gy [lindex $parts 1]
    set lname [lindex $parts 2]
    set demand $v_edge_demand($key)

    set pitch $layer_pitch($lname)
    set dir $layer_dir($lname)
    set capacity 0
    if {$pitch > 0} {
        if {$dir == "HORIZONTAL"} {
            set gcell_h [expr {[lindex $y_grids [expr {$gy + 1}]] - [lindex $y_grids $gy]}]
            set capacity [expr {$gcell_h / $pitch}]
        } else {
            set gcell_w [expr {[lindex $x_grids [expr {$gx + 1}]] - [lindex $x_grids $gx]}]
            set capacity [expr {$gcell_w / $pitch}]
        }
    }
    set overflow [expr {max(0, $demand - $capacity)}]
    if {!$first} { puts $fp "," }
    set first 0
    puts -nonewline $fp "    \{\"type\": \"V\", \"gx\": $gx, \"gy\": $gy, \"layer\": \"$lname\", \"capacity\": $capacity, \"demand\": $demand, \"overflow\": $overflow\}"
}

puts $fp ""
puts $fp "  \],"

puts $fp "  \"summary\": \{"
puts $fp "    \"guide_based\": \{"
puts $fp "      \"total_gcells_with_demand\": [llength [array names agg_cap]],"
puts $fp "      \"gcells_with_overflow\": $n_of,"
puts $fp "      \"total_overflow\": $total_of"
puts $fp "    \},"
puts $fp "    \"edge_based\": \{"
puts $fp "      \"total_h_edges\": [llength [array names h_edge_demand]],"
puts $fp "      \"total_v_edges\": [llength [array names v_edge_demand]],"
puts $fp "      \"h_demand\": $total_h_demand,"
puts $fp "      \"v_demand\": $total_v_demand,"
puts $fp "      \"h_capacity\": $total_h_capacity,"
puts $fp "      \"v_capacity\": $total_v_capacity,"
puts $fp "      \"total_overflow\": [expr {$total_h_overflow + $total_v_overflow}],"
puts $fp "      \"max_h_overflow\": $max_h_overflow,"
puts $fp "      \"max_v_overflow\": $max_v_overflow"
puts $fp "    \}"
puts $fp "  \}"
puts $fp "\}"
close $fp

puts "\nJSON written to $json_file"
puts "\n=== DONE ==="
