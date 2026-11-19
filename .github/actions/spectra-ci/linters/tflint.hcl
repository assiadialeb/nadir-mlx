# Spectra TFLint configuration (SPCTR-300): passed with --config, so the
# repository's .tflint.hcl is never read. Only the bundled terraform ruleset
# runs; no plugin is downloaded (tflint --init is never called).
plugin "terraform" {
  enabled = true
  preset  = "recommended"
}
