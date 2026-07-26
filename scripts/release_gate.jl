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
    required = (
        "Project.toml",
        "CITATION.cff",
        "CHANGELOG.md",
        "README.md",
        "LICENSE",
        "LICENSES/GPL-3.0-only.txt",
        "LICENSES/BSD-3-Clause.txt",
        "LICENSES/CC-BY-4.0.txt",
        "COPYRIGHT.md",
        "PROVENANCE.md",
        "THIRD_PARTY_NOTICES.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "REUSE.toml",
        "src/heom.jl",
        "src/hierarchy_pulses.jl",
        "docs/src/assets/example_figures/README.md",
        "docs/src/assets/model_code_generator_core.js",
        "docs/src/model_code_generator.md",
        "docs/src/releasing.md",
    )
    for filename in required
        path = joinpath(root, filename)
        isfile(path) ||
            error("release gate must run from the package checkout; missing $filename")
        filesize(path) > 0 || error("required release file is empty: $filename")
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
    read(joinpath(root, "LICENSES/GPL-3.0-only.txt"), String) == license ||
        error("LICENSES/GPL-3.0-only.txt must match the canonical root LICENSE")
    bsd_license = read(joinpath(root, "LICENSES/BSD-3-Clause.txt"), String)
    occursin("QuTiP developers and contributors", bsd_license) &&
        occursin("Redistribution and use in source and binary forms", bsd_license) &&
        occursin("THIS SOFTWARE IS PROVIDED", bsd_license) ||
        error("LICENSES/BSD-3-Clause.txt must retain the QuTiP BSD notice")
    cc_by_license = read(joinpath(root, "LICENSES/CC-BY-4.0.txt"), String)
    occursin("Creative Commons Attribution 4.0 International Public License",
            cc_by_license) &&
        occursin("Section 3 -- License Conditions", cc_by_license) ||
        error("LICENSES/CC-BY-4.0.txt must retain the complete CC BY 4.0 terms")

    reuse = TOML.parsefile(joinpath(root, "REUSE.toml"))
    get(reuse, "version", nothing) == 1 ||
        error("REUSE.toml must use schema version 1")
    annotations = get(reuse, "annotations", Any[])
    any(annotation ->
        get(annotation, "path", nothing) == "**" &&
        get(annotation, "precedence", nothing) == "aggregate" &&
        get(annotation, "SPDX-License-Identifier", nothing) ==
            "GPL-3.0-only" &&
        !isempty(strip(get(
            annotation, "SPDX-FileCopyrightText", ""))),
        annotations) ||
        error("REUSE.toml must aggregate the project GPL/copyright notice")

    copyright = read(joinpath(root, "COPYRIGHT.md"), String)
    occursin("GPL-3.0-only", copyright) ||
        error("COPYRIGHT.md must identify GPL-3.0-only")

    provenance = read(joinpath(root, "PROVENANCE.md"), String)
    occursin(r"(?i)\bHEOM\b", provenance) ||
        error("PROVENANCE.md must record the HEOM implementation provenance")
    occursin(r"(?i)\bPIQS\b", provenance) ||
        error("PROVENANCE.md must record the PIQS-inspired feature provenance")

    third_party = read(joinpath(root, "THIRD_PARTY_NOTICES.md"), String)
    occursin(r"(?i)\bQuTiP\b", third_party) ||
        error("THIRD_PARTY_NOTICES.md must retain the QuTiP notice")
    occursin("BSD-3-Clause", third_party) ||
        error("THIRD_PARTY_NOTICES.md must identify the QuTiP BSD-3-Clause license")
    occursin("CC-BY-4.0", third_party) ||
        error("THIRD_PARTY_NOTICES.md must identify the Platonic-sequence CC-BY-4.0 license")
    occursin("q-2025-03-12-1661", third_party) ||
        error("THIRD_PARTY_NOTICES.md must retain the Platonic-sequence source")

    # REUSE-IgnoreStart
    # These identifiers are expected contents of audited source snippets, not
    # licensing declarations for this release-gate file.
    heom_source = read(joinpath(root, "src/heom.jl"), String)
    occursin("# SPDX-SnippetBegin", heom_source) &&
        occursin("# SPDX-SnippetEnd", heom_source) &&
        occursin("SPDX-License-Identifier: BSD-3-Clause", heom_source) &&
        occursin("e5dbb0195fdbf37fb39d4e52e27c80594f8eb655",
            heom_source) ||
        error("src/heom.jl must retain the marked QuTiP BSD snippet")

    pulse_source = read(joinpath(root, "src/hierarchy_pulses.jl"), String)
    occursin("# SPDX-SnippetBegin", pulse_source) &&
        occursin("# SPDX-SnippetEnd", pulse_source) &&
        occursin("SPDX-License-Identifier: CC-BY-4.0", pulse_source) &&
        occursin("_TETRAHEDRAL_DD_WORD", pulse_source) &&
        occursin("_ICOSAHEDRAL_DD_WORD", pulse_source) ||
        error("src/hierarchy_pulses.jl must retain the marked CC BY pulse data")
    # REUSE-IgnoreEnd

    contributing = read(joinpath(root, "CONTRIBUTING.md"), String)
    occursin("GPL-3.0-only", contributing) ||
        error("CONTRIBUTING.md must state the inbound GPL-3.0-only license")
    occursin(r"(?i)\b(DCO|Developer Certificate of Origin)\b", contributing) ||
        error("CONTRIBUTING.md must document the DCO contribution process")
    occursin("Signed-off-by:", contributing) ||
        error("CONTRIBUTING.md must document the DCO Signed-off-by trailer")

    figure_provenance = read(
        joinpath(root, "docs/src/assets/example_figures/README.md"), String)
    occursin(r"(?i)\b(provenance|generated|generating)\b", figure_provenance) ||
        error("the curated-figure README must record figure provenance")

    generator = read(
        joinpath(root, "docs/src/assets/model_code_generator_core.js"), String)
    # REUSE-IgnoreStart
    # These strings describe generated Julia output, not this release-gate file.
    occursin(
        "# SPDX-FileCopyrightText: 2026 " *
        "PermutationalInvariantDynamics.jl contributors",
        generator) ||
        error("the model-code generator must emit a copyright SPDX header")
    occursin("# SPDX-License-Identifier: GPL-3.0-only", generator) ||
        error("the model-code generator must emit a GPL-3.0-only SPDX header")
    # REUSE-IgnoreEnd
    occursin("without an output-license exception", generator) ||
        error("the model-code generator must emit the no-exception notice")

    generator_docs = read(joinpath(root, "docs/src/model_code_generator.md"), String)
    occursin("GPL-3.0-only", generator_docs) &&
        occursin("without an exception for generated output", generator_docs) ||
        error("the model-code-generator guide must document generated-code licensing")

    workflow_directory = joinpath(root, ".github", "workflows")
    for filename in readdir(workflow_directory; join=true)
        (endswith(filename, ".yml") || endswith(filename, ".yaml")) ||
            continue
        source = read(filename, String)
        for matched in eachmatch(
                r"(?m)^\s*-?\s*uses:\s*([^#\s]+)", source)
            action = only(matched.captures)
            startswith(action, "./") && continue
            occursin(r"^[^@]+@[0-9a-f]{40}$", action) ||
                error("workflow action is not pinned to an immutable SHA: " *
                      "$action in $(basename(filename))")
        end
    end

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
        tag_commit = only(_capture_git(
            root, "rev-parse", "--verify", "$normalized^{commit}"))
        checkout_commit = only(_capture_git(
            root, "rev-parse", "--verify", "HEAD^{commit}"))
        tag_commit == checkout_commit ||
            error("checked-out commit $checkout_commit is not the target " *
                  "$tag_commit of tag $normalized")
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
