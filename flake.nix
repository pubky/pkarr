{
  description = "Flakebox Project template";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    flakebox.url = "github:rustshop/flakebox";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      flakebox,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        flakeboxLib = flakebox.lib.${system} {
          config = {
            github.ci.buildOutputs = [ ];
            flakebox.init.enable = false;
          };
        };

        buildPaths = [
          "Cargo.toml"
          "Cargo.lock"
          "pkarr"
          "server"
        ];

        buildSrc = flakeboxLib.filterSubPaths {
          root = builtins.path {
            name = "pkarr";
            path = ./.;
          };
          paths = buildPaths;
        };

        multiBuild = (flakeboxLib.craneMultiBuild { }) (
          craneLib':
          let
            craneLib = (
              craneLib'.overrideArgs {
                pname = "pkarr";
                src = buildSrc;
                nativeBuildInputs = [ ];
              }
            );
          in
          rec {

            workspaceDeps = craneLib.buildWorkspaceDepsOnly { };
            workspaceBuild = craneLib.buildWorkspace {
              cargoArtifacts = workspaceDeps;
            };
            "pkarr-server" =  craneLib.buildPackageGroup {
              packages = [ "pkarr-server" ];
              mainProgram = "pkarr-server";
            };
          }
        );
      in
      {
        packages = {
          pkarr-server = multiBuild.pkarr-server;
        };

        legacyPackages = multiBuild;

        devShells = flakeboxLib.mkShells { };
      }
    );
}
