{
  description = "ITG chart generation: local Qwen fine-tuning environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      runtimeLibraries = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.libsndfile
      ];
    in
    {
      formatter.${system} = pkgs.nixfmt;

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          gitMinimal
          gcc
          pkg-config
          ffmpeg-headless
          libsndfile
        ];

        # Use Nix's patched interpreter, never a downloaded FHS interpreter.
        UV_PYTHON = "${pkgs.python312}/bin/python3.12";
        UV_PYTHON_DOWNLOADS = "never";
        UV_LINK_MODE = "copy";
        PYTHONNOUSERSITE = "1";
        TOKENIZERS_PARALLELISM = "false";

        shellHook = ''
          # Python wheels need these shared libraries on NixOS. The NVIDIA
          # driver comes from the host; CUDA userspace libraries come from uv.
          export LD_LIBRARY_PATH="${runtimeLibraries}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          if [ -d /run/opengl-driver/lib ]; then
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          fi
          echo "ITG environment: bash scripts/setup.sh cpu (development) or cuda (NVIDIA training)."
        '';
      };
    };
}
