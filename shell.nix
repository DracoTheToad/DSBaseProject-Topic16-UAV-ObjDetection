
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
  ];

  PYTHONPATH = toString ./.;
  # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=0.70;

  shellHook = ''
    echo "Hi :>"
  '';
}

