using BaytesFilters
using Documenter

DocMeta.setdocmeta!(BaytesFilters, :DocTestSetup, :(using BaytesFilters); recursive=true)

makedocs(;
    modules=[BaytesFilters],
    authors="Patrick Aschermayr <p.aschermayr@gmail.com>",
    repo="https://github.com/paschermayr/BaytesFilters.jl/blob/{commit}{path}#{line}",
    sitename="BaytesFilters.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://paschermayr.github.io/BaytesFilters.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/paschermayr/BaytesFilters.jl",
    devbranch="master",
)
