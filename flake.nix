{
  description = "nn in rust";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
  };

  outputs = { self, nixpkgs }: 
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
    };
  in {
    packages.${system}.default = pkgs.rustPlatform.buildRustPackage {
      name = "rs-mnist";
      version = "1.0.0";
      cargoLock = {
        lockFile = ./Cargo.lock;
      };
      src = self;
      # buildInputs = [ pkgs.rustc pkgs.cargo ];
    };

    devShells.${system}.default = pkgs.mkShell {
      packages = [
        pkgs.rustc
        pkgs.cargo
        pkgs.rust-analyzer
        pkgs.lldb_9
      ];
    shellHook = ''
      echo "ENTERING DEV"
      $SHELL
      echo "EXITING DEV"
      exit
    '';
    };
  };
}
