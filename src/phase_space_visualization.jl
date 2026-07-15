"""
    SpinPhaseSpaceVisualization

Dependency-free equirectangular SVG rendering configuration for a
[`SpinPhaseSpaceData`](@ref) object. `sector_index == 0` renders its aggregate
Schur-sector marginal; a positive index renders one retained sector matrix.
Rendering changes no numerical values and never recomputes a phase-space
transform.
"""
struct SpinPhaseSpaceVisualization{D,L}
    data::D
    sector_index::Int
    title::String
    width::Int
    height::Int
    colorlimits::L
    palette::Symbol
    show_colorbar::Bool
end

const _SPIN_Q_PALETTE=(
    "#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6",
    "#4292c6","#2171b5","#08519c","#08306b")

const _SPIN_WIGNER_PALETTE=(
    "#053061","#2166ac","#4393c3","#92c5de","#d1e5f0",
    "#f7f7f7","#fddbc7","#f4a582","#d6604d","#b2182b","#67001f")

function _spin_visual_sector_index(data::SpinPhaseSpaceData,sector)
    sector===:aggregate&&return 0
    data.sector_values===nothing&&throw(ArgumentError(
        "sector rendering requires data computed with resolved=true"))
    partition=if sector isa Partition
        length(sector)==2||throw(ArgumentError("spin-sector labels must have length 2"))
        sector
    elseif (sector isa Tuple||sector isa AbstractVector)&&length(sector)==2&&
           all(value->value isa Integer,sector)
        Partition((Int(sector[1]),Int(sector[2])))
    else
        throw(ArgumentError(
            "sector must be :aggregate or a retained two-row partition"))
    end
    index=findfirst(==(partition),data.sectors)
    index===nothing&&throw(ArgumentError(
        "sector $partition is not retained in this phase-space result"))
    index
end

function _spin_visual_matrix(data,index)
    index==0 ? data.values : data.sector_values[index]
end

function _strict_regular_grid(values,name)
    length(values)>=2||throw(ArgumentError(
        "$name needs at least two samples for SVG visualization"))
    all(diff(values).>zero(eltype(values)))||throw(ArgumentError(
        "$name samples must be strictly increasing for SVG visualization"))
    steps=diff(values);reference=first(steps)
    tolerance=sqrt(eps(eltype(values)))*max(abs(reference),one(reference))
    all(step->abs(step-reference)<=tolerance,steps)||throw(ArgumentError(
        "$name samples must form a regular grid for SVG visualization"))
    nothing
end

function _spin_visual_colorlimits(values,kind,colorlimits)
    minimum_value=minimum(values);maximum_value=maximum(values)
    if colorlimits===nothing
        if kind===:wigner
            bound=max(abs(minimum_value),abs(maximum_value))
            if iszero(bound)
                bound=one(bound)
            end
            return (-bound,bound)
        end
        if minimum_value==maximum_value
            width=max(abs(minimum_value),one(minimum_value))*
                  sqrt(eps(typeof(minimum_value)))
            return (minimum_value-width,maximum_value+width)
        end
        return (minimum_value,maximum_value)
    end
    colorlimits isa Tuple&&length(colorlimits)==2||throw(ArgumentError(
        "colorlimits must be nothing or a (lower, upper) tuple"))
    lower,upper=colorlimits
    lower isa Real&&upper isa Real||throw(ArgumentError(
        "color limits must be real"))
    isfinite(lower)&&isfinite(upper)&&lower<upper||throw(ArgumentError(
        "color limits must be finite and strictly increasing"))
    R=eltype(values)
    converted=(R(lower),R(upper))
    all(isfinite,converted)&&converted[1]<converted[2]||throw(ArgumentError(
        "color limits are not representable in $R"))
    converted
end

function _spin_visual_palette(kind,palette)
    resolved=palette===:auto ? (kind===:wigner ? :diverging : :sequential) : palette
    resolved in (:sequential,:diverging)||throw(ArgumentError(
        "palette must be :auto, :sequential, or :diverging"))
    resolved
end

"""
    visualize_spin_phase_space(data; sector=:aggregate, title=nothing,
                               width=760, height=460,
                               colorlimits=nothing, palette=:auto,
                               show_colorbar=true)

Create a dependency-free equirectangular heatmap of precomputed spin
phase-space data. Azimuth `phi` runs horizontally and polar angle `theta`
runs vertically with the north pole at the top. Both coordinate vectors must
be strictly increasing regular grids; numerical evaluation itself permits
arbitrary grids.

The default renders the aggregate marginal. Pass a retained partition to
`sector` after computing the data with `resolved=true` to display one
sector-resolved sphere. Husimi-Q data use a sequential palette and Wigner data
use a zero-centered diverging palette by default. `colorlimits` affects only
presentation: underlying values, including Wigner negativity and values
outside the displayed color range, are unchanged.
"""
function visualize_spin_phase_space(data::SpinPhaseSpaceData;
        sector=:aggregate,title=nothing,width::Integer=760,height::Integer=460,
        colorlimits=nothing,palette::Symbol=:auto,show_colorbar::Bool=true)
    data.kind in (:husimi_q,:wigner)||throw(ArgumentError(
        "unsupported spin phase-space kind $(data.kind)"))
    _strict_regular_grid(data.theta,"theta")
    _strict_regular_grid(data.phi,"phi")
    big(length(data.theta))*big(length(data.phi))<=100_000||throw(ArgumentError(
        "phase-space SVG grid exceeds 100000 cells; render a smaller sampled grid"))
    width>=420||throw(ArgumentError("width must be at least 420"))
    height>=320||throw(ArgumentError("height must be at least 320"))
    width<=typemax(Int)&&height<=typemax(Int)||throw(ArgumentError(
        "visualization dimensions are too large"))
    sector_index=_spin_visual_sector_index(data,sector)
    values=_spin_visual_matrix(data,sector_index)
    all(isfinite,values)||throw(ArgumentError(
        "phase-space values must be finite for SVG rendering"))
    limits=_spin_visual_colorlimits(values,data.kind,colorlimits)
    resolved_palette=_spin_visual_palette(data.kind,palette)
    default_title=data.kind===:husimi_q ? "Spin Husimi-Q distribution" :
                                         "Spin Wigner distribution"
    SpinPhaseSpaceVisualization(data,sector_index,
        title===nothing ? default_title : string(title),Int(width),Int(height),
        limits,resolved_palette,show_colorbar)
end

function show(io::IO,visualization::SpinPhaseSpaceVisualization)
    label=visualization.sector_index==0 ? "aggregate" :
          string(visualization.data.sectors[visualization.sector_index])
    print(io,"SpinPhaseSpaceVisualization(kind=",visualization.data.kind,
          ", sector=",label,", size=",visualization.width,"×",
          visualization.height,")")
end

function show(io::IO,::MIME"text/plain",
              visualization::SpinPhaseSpaceVisualization)
    show(io,visualization)
    print(io,"\n  title: ",repr(visualization.title),
          "\n  projection: equirectangular",
          "\n  use display(...) for SVG output")
end

function _spin_phase_palette(visualization)
    visualization.palette===:sequential ? _SPIN_Q_PALETTE :
                                          _SPIN_WIGNER_PALETTE
end

function _spin_phase_color_index(value,limits,count)
    lower,upper=limits
    normalized=Float64((value-lower)/(upper-lower))
    isfinite(normalized)||throw(ArgumentError(
        "phase-space color coordinate is not representable for SVG rendering"))
    clamp(floor(Int,clamp(normalized,0.0,1.0)*(count-1))+1,1,count)
end

_spin_phase_number(value)=string(round(value;sigdigits=5))

function show(io::IO,::MIME"image/svg+xml",
              visualization::SpinPhaseSpaceVisualization)
    data=visualization.data
    values=_spin_visual_matrix(data,visualization.sector_index)
    palette=_spin_phase_palette(visualization)
    paths=[IOBuffer() for _ in palette]
    width=visualization.width;height=visualization.height
    left=76.0;right=visualization.show_colorbar ? 112.0 : 34.0
    top=76.0;bottom=62.0
    plot_width=width-left-right;plot_height=height-top-bottom
    phi_count=length(data.phi);theta_count=length(data.theta)
    cell_width=plot_width/phi_count;cell_height=plot_height/theta_count
    clipped=0
    @inbounds for theta_index in 1:theta_count,phi_index in 1:phi_count
        value=values[phi_index,theta_index]
        (value<visualization.colorlimits[1]||value>visualization.colorlimits[2])&&
            (clipped+=1)
        color_index=_spin_phase_color_index(value,visualization.colorlimits,
                                             length(palette))
        x=left+(phi_index-1)*cell_width
        y=top+(theta_index-1)*cell_height
        print(paths[color_index],"M",round(x;digits=3)," ",round(y;digits=3),
              "h",round(cell_width+0.02;digits=3),
              "v",round(cell_height+0.02;digits=3),
              "h-",round(cell_width+0.02;digits=3),"z")
    end
    sector_label=if visualization.sector_index==0
        "aggregate marginal over $(length(data.sectors)) Schur sectors"
    else
        index=visualization.sector_index
        partition=data.sectors[index]
        "sector $partition, j=$(data.twice_spins[index])/2, population=$(_spin_phase_number(data.populations[index]))"
    end
    print(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"",width,
          "\" height=\"",height,"\" viewBox=\"0 0 ",width," ",height,
          "\" role=\"img\"><title>",_svg_escape(visualization.title),
          "</title><desc>",_svg_escape(sector_label),
          "; values[phi_index, theta_index]; sphere-density normalization</desc>",
          "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
          "<text x=\"",width/2,"\" y=\"28\" text-anchor=\"middle\" ",
          "font-family=\"sans-serif\" font-size=\"18\" font-weight=\"600\">",
          _svg_escape(visualization.title),"</text>",
          "<text x=\"",width/2,"\" y=\"49\" text-anchor=\"middle\" ",
          "font-family=\"sans-serif\" font-size=\"11\" fill=\"#475569\">",
          _svg_escape(sector_label),"</text>")
    for index in eachindex(paths)
        position(paths[index])==0&&continue
        print(io,"<path d=\"",String(take!(paths[index])),"\" fill=\"",
              palette[index],"\" stroke=\"none\"/>")
    end
    print(io,"<rect x=\"",left,"\" y=\"",top,"\" width=\"",plot_width,
          "\" height=\"",plot_height,
          "\" fill=\"none\" stroke=\"#334155\" stroke-width=\"1\"/>",
          "<text x=\"",left+plot_width/2,"\" y=\"",height-25,
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">φ</text>",
          "<text x=\"",left-48,"\" y=\"",top+plot_height/2,
          "\" text-anchor=\"middle\" transform=\"rotate(-90 ",left-48," ",
          top+plot_height/2,")\" font-family=\"sans-serif\" font-size=\"12\">θ</text>")
    for (x,label) in ((left,_spin_phase_number(first(data.phi))),
                      (left+plot_width,_spin_phase_number(last(data.phi))))
        print(io,"<text x=\"",x,"\" y=\"",top+plot_height+18,
              "\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"9\">",
              _svg_escape(label),"</text>")
    end
    for (y,label) in ((top,_spin_phase_number(first(data.theta))),
                      (top+plot_height,_spin_phase_number(last(data.theta))))
        print(io,"<text x=\"",left-8,"\" y=\"",y+3,
              "\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"9\">",
              _svg_escape(label),"</text>")
    end
    if visualization.show_colorbar
        bar_x=left+plot_width+34;bar_y=top+12
        bar_height=plot_height-24;segment=bar_height/length(palette)
        for index in eachindex(palette)
            # Highest palette entry is placed at the top.
            y=bar_y+(length(palette)-index)*segment
            print(io,"<rect x=\"",bar_x,"\" y=\"",y,"\" width=\"18\" height=\"",
                  segment+0.02,"\" fill=\"",palette[index],"\"/>")
        end
        print(io,"<rect x=\"",bar_x,"\" y=\"",bar_y,"\" width=\"18\" height=\"",
              bar_height,"\" fill=\"none\" stroke=\"#334155\" stroke-width=\"0.8\"/>",
              "<text x=\"",bar_x+24,"\" y=\"",bar_y+4,
              "\" font-family=\"monospace\" font-size=\"9\">",
              _svg_escape(_spin_phase_number(visualization.colorlimits[2])),"</text>",
              "<text x=\"",bar_x+24,"\" y=\"",bar_y+bar_height+3,
              "\" font-family=\"monospace\" font-size=\"9\">",
              _svg_escape(_spin_phase_number(visualization.colorlimits[1])),"</text>")
    end
    footer="$(data.normalization) · equirectangular"
    clipped==0||(footer*=" · $clipped colors clipped")
    print(io,"<text x=\"",width/2,"\" y=\"",height-8,
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"9\" fill=\"#475569\">",
          _svg_escape(footer),"</text></svg>")
end

"""Write a spin phase-space SVG and return the path as a `String`."""
function save_spin_phase_space_visualization(
        path::AbstractString,visualization::SpinPhaseSpaceVisualization)
    svg=sprint(show,MIME"image/svg+xml"(),visualization)
    open(path,"w") do io
        write(io,svg)
    end
    String(path)
end

function save_spin_phase_space_visualization(
        path::AbstractString,data::SpinPhaseSpaceData;kwargs...)
    save_spin_phase_space_visualization(
        path,visualize_spin_phase_space(data;kwargs...))
end
