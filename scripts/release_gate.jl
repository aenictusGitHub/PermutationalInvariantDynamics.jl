module ReleaseGate

using Dates
using TOML

const REPOSITORY =
    "https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl"
const DOCUMENTATION =
    "https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl"
const PACKAGE_UUID = "8edbc7f6-5364-4b2e-9e03-73a6d67b7c8a"

function _capture_git(root::AbstractString, arguments::AbstractString...)
    command = Cmd(vcat(["git", "-C", root], collect(arguments)))
    split(chomp(read(command, String)), '\n'; keepempty=false)
end

function _first_capture(pattern::Regex, text::AbstractString, label::AbstractString)
    match_result = match(pattern, text)
    match_result === nothing && error("missing $label")
    only(match_result.captures)
end

function _release_date(changelog::AbstractString, version::VersionNumber)
    escaped = replace(string(version), "." => raw"\.")
    value = _first_capture(
        Regex("(?m)^## \\[$escaped\\] - (Unreleased|\\d{4}-\\d{2}-\\d{2})\\s*\$"),
        changelog,
        "CHANGELOG.md heading for version $version",
    )
    value == "Unreleased" ? nothing : Date(value)
end

"""
    release_gate(root=pwd(); require_clean=false, require_released=false,
                 expected_version=nothing, tag_ref=nothing)

Validate the local metadata and repository-hygiene part of the package release
gate. The default accepts a synchronized release candidate whose changelog is
still marked `Unreleased`. Set `require_released=true` only after assigning the
release date in both `CHANGELOG.md` and `CITATION.cff`.

This function does not create a tag, GitHub release, or General-registry pull
request.
"""
function release_gate(
        root::AbstractString=pwd();
        require_clean::Bool=false,
        require_released::Bool=false,
        expected_version::Union{Nothing,VersionNumber}=nothing,
        tag_ref::Union{Nothing,AbstractString}=nothing)
    root = abspath(root)
    required = ("Project.toml", "CITATION.cff", "CHANGELOG.md", "README.md",
                "LICENSE", "docs/src/releasing.md")
    for filename in required
        isfile(joinpath(root, filename)) ||
            error("release gate must run from the package checkout; missing $filename")
    end

    project = TOML.parsefile(joinpath(root, "Project.toml"))
    get(project, "name", nothing) == "PermutationalInvariantDynamics" ||
        error("unexpected package name in Project.toml")
    get(project, "uuid", nothing) == PACKAGE_UUID ||
        error("unexpected package UUID in Project.toml")
    haskey(project, "version") || error("Project.toml has no version")
    version = VersionNumber(project["version"])
    expected_version === nothing || version == expected_version ||
        error("Project.toml version $version does not match requested " *
              "$expected_version")

    citation = read(joinpath(root, "CITATION.cff"), String)
    citation_version = VersionNumber(_first_capture(
        r"(?m)^version:\s*[\"']?([^\"'\s]+)[\"']?\s*$",
        citation,
        "CITATION.cff version",
    ))
    citation_version == version ||
        error("CITATION.cff version $citation_version does not match " *
              "Project.toml version $version")
    occursin("license: \"GPL-3.0-only\"", citation) ||
        error("CITATION.cff must declare GPL-3.0-only")
    occursin("repository-code: \"$REPOSITORY\"", citation) ||
        error("CITATION.cff repository-code does not match the public repository")

    changelog = read(joinpath(root, "CHANGELOG.md"), String)
    release_date = _release_date(changelog, version)
    release_link = "[$version]: $REPOSITORY/releases/tag/v$version"
    occursin(release_link, changelog) ||
        error("CHANGELOG.md must contain the canonical release link $release_link")
    citation_date_match = match(
        r"(?m)^date-released:\s*[\"']?(\d{4}-\d{2}-\d{2})[\"']?\s*$",
        citation,
    )
    citation_date = citation_date_match === nothing ? nothing :
        Date(only(citation_date_match.captures))
    if release_date === nothing
        citation_date === nothing ||
            error("CITATION.cff has date-released=$citation_date while " *
                  "CHANGELOG.md still says Unreleased")
        require_released && error(
            "version $version is still marked Unreleased")
    else
        citation_date == release_date ||
            error("CITATION.cff date-released $(something(citation_date, "missing")) " *
                  "does not match CHANGELOG.md date $release_date")
    end

    readme = read(joinpath(root, "README.md"), String)
    occursin(REPOSITORY, readme) ||
        error("README.md does not reference the public repository")
    occursin(DOCUMENTATION, readme) ||
        error("README.md does not reference the hosted documentation")
    occursin("GPL-3.0-only", readme) ||
        error("README.md does not identify the GPL-3.0-only license")
    occursin("Codex", readme) ||
        error("README.md must retain the substantial Codex-assistance disclosure")

    license = read(joinpath(root, "LICENSE"), String)
    occursin("GNU GENERAL PUBLIC LICENSE", license) &&
        occursin("Version 3, 29 June 2007", license) ||
        error("LICENSE is not the canonical GPL version 3 text")

    tracked_manifests = _capture_git(
        root, "ls-files", "--", ":(glob)**/Manifest.toml")
    tracked_manifests == ["docs/Manifest.toml"] ||
        error("only docs/Manifest.toml may be tracked; found " *
              repr(tracked_manifests))

    tracked_generated = _capture_git(
        root, "ls-files", "--", "docs/build", "examples/figures")
    isempty(tracked_generated) ||
        error("generated documentation/example output is tracked: " *
              join(tracked_generated, ", "))

    if require_clean
        dirty = _capture_git(root, "status", "--porcelain",
                             "--untracked-files=all")
        isempty(dirty) ||
            error("nonignored worktree changes remain:\n" * join(dirty, "\n"))
    end

    if tag_ref !== nothing
        normalized = replace(String(tag_ref), r"^refs/tags/" => "")
        normalized == "v$version" ||
            error("tag $normalized does not match package version v$version")
        release_date === nothing &&
            error("tag validation requires a dated release")
    end

    (
        package=String(project["name"]),
        version=version,
        status=release_date === nothing ? :candidate : :released,
        release_date=release_date,
        tracked_manifest=only(tracked_manifests),
        clean=require_clean,
    )
end

function _usage(io::IO=stdout)
    print(io, """
Usage: julia scripts/release_gate.jl [options]

Options:
  --root PATH              package checkout (default: repository parent)
  --expect-version X.Y.Z   require this Project.toml version
  --require-clean          reject tracked worktree modifications
  --require-released       require matching release dates in changelog/citation
  --tag-ref REF            require REF to equal vX.Y.Z (also requires a date)
  --help                   show this message
""")
end

function main(arguments::Vector{String}=ARGS)
    root = normpath(joinpath(@__DIR__, ".."))
    expected_version = nothing
    tag_ref = nothing
    require_clean = false
    require_released = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--root"
            index == length(arguments) && error("--root needs a value")
            index += 1
            root = arguments[index]
        elseif argument == "--expect-version"
            index == length(arguments) &&
                error("--expect-version needs a value")
            index += 1
            expected_version = VersionNumber(arguments[index])
        elseif argument == "--tag-ref"
            index == length(arguments) && error("--tag-ref needs a value")
            index += 1
            tag_ref = arguments[index]
        elseif argument == "--require-clean"
            require_clean = true
        elseif argument == "--require-released"
            require_released = true
        elseif argument in ("-h", "--help")
            _usage()
            return nothing
        else
            error("unknown option $argument; use --help")
        end
        index += 1
    end

    report = release_gate(
        root;
        require_clean=require_clean,
        require_released=require_released,
        expected_version=expected_version,
        tag_ref=tag_ref,
    )
    println("release metadata gate passed")
    println("  package: ", report.package)
    println("  version: ", report.version)
    println("  status: ", report.status)
    println("  release date: ", something(report.release_date, "not assigned"))
    println("  tracked manifest: ", report.tracked_manifest)
    println("  clean worktree required: ", report.clean)
    nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()

end
