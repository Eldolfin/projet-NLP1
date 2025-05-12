{
  inputs = {
    utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    self,
    nixpkgs,
    utils,
  }:
    utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            lolcat
            (
              python313.withPackages
              (
                ppkgs:
                  with ppkgs; [
                    tiktoken
                    sentence-transformers
                    ipython
                    rich
                    textual
                    datasets
                    jupyter-all
                    notebook
                    nltk
                    numpy
                    snakeviz
                    python-lsp-server
                    python-lsp-ruff
                    pylsp-rope
                    pylsp-mypy
                    pyls-isort

                    accelerate
                    black
                    graphviz
                    ipdb
                    isort
                    jupyterlab
                    kneed
                    matplotlib
                    numpy
                    opencv-python
                    optuna
                    pandas
                    pudb
                    scikit-learn
                    scipy
                    seaborn
                    snakeviz
                    termcolor
                    transformers
                    [torch]
                    torch
                    wikipedia
                  ]
              )
            )
          ];
        };
      }
    );
}
