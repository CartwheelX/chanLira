#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '[INFO] satml_noise_budget.sh now invokes the corrected N2 API-query study.'
printf '%s\n' '[INFO] Capture and export one frozen snapshot before running this command.'
exec bash commands/satml_noise_n2_queries.sh
