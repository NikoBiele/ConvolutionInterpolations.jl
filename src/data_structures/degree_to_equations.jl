"""
    DEGREE_TO_EQUATIONS

Maps kernel degree symbols to the number of piecewise equations (= support radius).

Keys are kernel symbols (`:a0`, `:a1`, `:a3`, `:a4`, `:a5`, `:a7`, `:b5`, `:b7`, `:b9`,
`:b11`, `:b13`). The support range of kernel `:k` is `[-E, E]` where
`E = DEGREE_TO_EQUATIONS[:k]`.

Used internally by `get_equations_for_degree`.
"""

const DEGREE_TO_EQUATIONS = Dict(
    :a0 => 1,
    :a1 => 1,
    :a3 => 2,
    :a4 => 3,
    :a5 => 3,
    :a7 => 4,
    :b5 => 5,
    :b7 => 6,
    :b9 => 7,
    :b11 => 8,
    :b13 => 9
)