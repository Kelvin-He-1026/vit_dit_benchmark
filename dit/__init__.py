"""DiT text-to-image benchmarks.

  dit_benchmark         end-to-end generation, or offline throughput sweep
                        (--throughput)
  server_dit_benchmark  request-rate ramp against a latency SLA
  catalog               model list, load caveats and prompt loader shared by
                        the two
  run_server_dit_sweep.sh  server_dit_benchmark over hardware x model x
                           precision

Run from the repository root, e.g. python -m dit.dit_benchmark --help
"""
