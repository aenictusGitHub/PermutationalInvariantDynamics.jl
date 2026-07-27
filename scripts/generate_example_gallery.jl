module ExampleGalleryBuilder

using TOML

const REPOSITORY_URL =
    "https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl"

function _escape_html(value)
    replace(
        string(value),
        '&'=>"&amp;",
        '<'=>"&lt;",
        '>'=>"&gt;",
        '"'=>"&quot;",
        '\''=>"&#39;",
    )
end

function render(catalog_path::AbstractString)
    catalog=TOML.parsefile(catalog_path)
    get(catalog,"schema_version",nothing)==1||
        error("unsupported examples catalog schema")
    entries=get(catalog,"example",nothing)
    entries isa Vector||error("examples catalog has no example array")
    sort!(entries;by=entry->lowercase(entry["title"]))

    tasks=sort!(unique(String[
        task for entry in entries for task in entry["tasks"]]))
    output=IOBuffer()
    println(output,"# Searchable example gallery")
    println(output)
    println(output,
        "This page is generated from [`examples/catalog.toml`](",
        REPOSITORY_URL,"/blob/main/examples/catalog.toml). ",
        "Every card links to a runnable script, its physical guide, and a ",
        "reviewed expected output. Filters run locally in the browser.")
    println(output)
    println(output,"```@raw html")
    println(output,"<style>")
    println(output,
        ".pid-example-filters{display:grid;grid-template-columns:",
        "minmax(14rem,2fr) repeat(3,minmax(9rem,1fr));gap:.75rem;",
        "align-items:end;margin:1rem 0}.pid-example-filters label{display:",
        "grid;gap:.25rem;font-weight:600}.pid-example-filters input,",
        ".pid-example-filters select{width:100%;padding:.5rem;border:1px ",
        "solid #8a8f98;border-radius:.4rem;background:inherit;color:inherit}",
        ".pid-example-cards{display:grid;grid-template-columns:",
        "repeat(auto-fit,minmax(18rem,1fr));gap:1rem}.pid-example-card{",
        "border:1px solid #8a8f9866;border-radius:.65rem;padding:.9rem;",
        "min-width:0}.pid-example-card[hidden]{display:none}",
        ".pid-example-card img{width:100%;height:11rem;object-fit:contain;",
        "border-radius:.35rem;background:#fff}.pid-example-card h2{",
        "font-size:1.08rem;margin:.65rem 0 .35rem}.pid-example-card p{",
        "margin:.35rem 0}.pid-example-card pre{overflow:auto;font-size:",
        ".75rem}.pid-example-tasks code{white-space:nowrap}",
        "@media(max-width:52rem){.pid-example-filters{grid-template-columns:",
        "1fr 1fr}}@media(max-width:32rem){.pid-example-filters{",
        "grid-template-columns:1fr}}")
    println(output,"</style>")
    println(output,"<div id=\"pid-example-gallery\" class=\"pid-example-gallery\">")
    println(output,"  <div class=\"pid-example-filters\">")
    println(output,
        "    <label>Search <input id=\"pid-example-search\" type=\"search\" ",
        "placeholder=\"model, task, citation…\"></label>")
    println(output,"    <label>Difficulty <select id=\"pid-example-difficulty\">")
    println(output,"      <option value=\"\">All</option>")
    for difficulty in ("beginner","intermediate","advanced")
        println(output,"      <option value=\"",difficulty,"\">",
            uppercasefirst(difficulty),"</option>")
    end
    println(output,"    </select></label>")
    println(output,"    <label>Task <select id=\"pid-example-task\">")
    println(output,"      <option value=\"\">All</option>")
    for task in tasks
        println(output,"      <option value=\"",_escape_html(task),"\">",
            _escape_html(task),"</option>")
    end
    println(output,"    </select></label>")
    println(output,"    <label>Method <select id=\"pid-example-stochastic\">")
    println(output,"      <option value=\"\">All</option>")
    println(output,"      <option value=\"false\">Deterministic</option>")
    println(output,"      <option value=\"true\">Stochastic</option>")
    println(output,"    </select></label>")
    println(output,"  </div>")
    println(output,"  <p id=\"pid-example-count\" role=\"status\"></p>")
    println(output,"  <div class=\"pid-example-cards\">")
    for entry in entries
        tasks_text=join(entry["tasks"]," ")
        searchable=lowercase(join((
            entry["title"],entry["citation"],entry["difficulty"],
            entry["runtime_class"],tasks_text,
        )," "))
        script=entry["script"]
        guide=entry["guide"]
        preview=first(entry["expected_outputs"])
        preview_relative=replace(preview,"docs/src/"=>"")
        println(output,
            "    <article class=\"pid-example-card\" data-search=\"",
            _escape_html(searchable),"\" data-difficulty=\"",
            _escape_html(entry["difficulty"]),"\" data-tasks=\"",
            _escape_html(tasks_text),"\" data-stochastic=\"",
            entry["stochastic"],"\">")
        println(output,"      <img loading=\"lazy\" src=\"../",
            _escape_html(preview_relative),"\" alt=\"Expected output for ",
            _escape_html(entry["title"]),"\">")
        println(output,"      <h2>",_escape_html(entry["title"]),"</h2>")
        println(output,"      <p><strong>",
            _escape_html(entry["difficulty"]),"</strong> · ",
            _escape_html(entry["runtime_class"])," · ",
            entry["stochastic"] ? "stochastic" : "deterministic","</p>")
        println(output,"      <p>",_escape_html(entry["citation"]),"</p>")
        println(output,"      <p class=\"pid-example-tasks\">",
            join(("<code>"*_escape_html(task)*"</code>"
                  for task in entry["tasks"])," "),"</p>")
        println(output,"      <p><a href=\"",REPOSITORY_URL,"/blob/main/",
            _escape_html(guide),"\">Guide</a> · <a href=\"",
            REPOSITORY_URL,"/blob/main/",_escape_html(script),
            "\">Script</a></p>")
        println(output,"      <pre><code>julia --project=. ",
            _escape_html(script),"</code></pre>")
        println(output,"    </article>")
    end
    println(output,"  </div>")
    println(output,"</div>")
    println(output,"<script>")
    println(output,"(() => {")
    println(output,
        "  const root = document.getElementById('pid-example-gallery');")
    println(output,"  if (!root) return;")
    println(output,
        "  const cards = Array.from(root.querySelectorAll('.pid-example-card'));")
    println(output,"  const search = root.querySelector('#pid-example-search');")
    println(output,
        "  const difficulty = root.querySelector('#pid-example-difficulty');")
    println(output,"  const task = root.querySelector('#pid-example-task');")
    println(output,
        "  const stochastic = root.querySelector('#pid-example-stochastic');")
    println(output,"  const count = root.querySelector('#pid-example-count');")
    println(output,"  function filter() {")
    println(output,"    const query = search.value.trim().toLowerCase();")
    println(output,"    let visible = 0;")
    println(output,"    for (const card of cards) {")
    println(output,
        "      const tasks = card.dataset.tasks.split(/\\s+/);")
    println(output,"      const keep = (!query || card.dataset.search.includes(query)) &&")
    println(output,
        "        (!difficulty.value || card.dataset.difficulty === difficulty.value) &&")
    println(output,
        "        (!task.value || tasks.includes(task.value)) &&")
    println(output,
        "        (!stochastic.value || card.dataset.stochastic === stochastic.value);")
    println(output,"      card.hidden = !keep;")
    println(output,"      if (keep) visible += 1;")
    println(output,"    }")
    println(output,
        "    count.textContent = `\${visible} of \${cards.length} examples`;")
    println(output,"  }")
    println(output,
        "  for (const control of [search, difficulty, task, stochastic]) {")
    println(output,"    control.addEventListener('input', filter);")
    println(output,"    control.addEventListener('change', filter);")
    println(output,"  }")
    println(output,"  filter();")
    println(output,"})();")
    println(output,"</script>")
    println(output,"```")
    String(take!(output))
end

function main(arguments=ARGS)
    root=normpath(joinpath(@__DIR__,".."))
    catalog=isempty(arguments) ?
        joinpath(root,"examples","catalog.toml") : abspath(arguments[1])
    destination=length(arguments)<2 ?
        joinpath(root,"docs","src","example_gallery.md") :
        abspath(arguments[2])
    write(destination,render(catalog))
    println(destination)
    destination
end

end

abspath(PROGRAM_FILE)==abspath(@__FILE__)&&
    ExampleGalleryBuilder.main()
