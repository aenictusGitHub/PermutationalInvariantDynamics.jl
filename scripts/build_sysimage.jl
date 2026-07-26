module BuildPIDSysimage

function usage(io::IO=stdout)
    print(io,"""
Build a local PermutationalInvariantDynamics sysimage.

Usage:
  julia --project=/path/to/sysimage-env scripts/build_sysimage.jl [options]

Options:
  --output PATH     output library path (default: platform-specific file in /tmp)
  --project PATH    package project to bake into the image (default: repository)
  --cpu-target NAME PackageCompiler cpu_target (default: native)
  --help            show this message

The active environment must contain PackageCompiler and this local package,
for example:

  julia --project=/tmp/pid-sysimage -e 'using Pkg; \\
      Pkg.add("PackageCompiler"); Pkg.develop(path=pwd())'
""")
end

function default_output()
    extension=Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so"
    joinpath(tempdir(),"PermutationalInvariantDynamics$extension")
end

function options(arguments)
    output=default_output()
    project=normpath(joinpath(@__DIR__,".."))
    cpu_target="native"
    index=1
    while index<=length(arguments)
        argument=arguments[index]
        if argument=="--help"
            usage()
            return nothing
        elseif argument in ("--output","--project","--cpu-target")
            index<length(arguments)||error("$argument requires a value")
            value=arguments[index+1]
            argument=="--output" ? (output=abspath(value)) :
            argument=="--project" ? (project=abspath(value)) :
            (cpu_target=value)
            index+=2
        else
            error("unknown option $argument; use --help")
        end
    end
    (;output,project,cpu_target)
end

function main(arguments=ARGS)
    configuration=options(arguments)
    configuration===nothing&&return nothing
    Base.find_package("PackageCompiler")===nothing&&error(
        "PackageCompiler is not available in the active environment; "*
        "install it in a separate sysimage environment as shown by --help")
    @eval import PackageCompiler
    workload=joinpath(@__DIR__,"precompile_workload.jl")
    isfile(joinpath(configuration.project,"Project.toml"))||error(
        "project does not contain Project.toml: $(configuration.project)")
    mkpath(dirname(configuration.output))
    PackageCompiler.create_sysimage(
        [:PermutationalInvariantDynamics];
        project=configuration.project,
        sysimage_path=configuration.output,
        precompile_execution_file=workload,
        cpu_target=configuration.cpu_target,
    )
    println(configuration.output)
    configuration.output
end

end

abspath(PROGRAM_FILE)==abspath(@__FILE__)&&BuildPIDSysimage.main()
