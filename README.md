# nipypeVBM

This is a nipype implementation of the FSL VBM scripts. Modifications to the pipeline include replacing the FSL FLIRT/FNIRT registrations with ANTS Registration and replacing the FSL FAST tissue classifications with ANTS Atropos (3 tissue classes, using mni_icbm152_nlin_sym_09c atlas as priors). Everthing else tries to faithfully reproduce the FSL VBM scripts, including splitting them into three scripts - fslvbm_1_bet, fslvbm_2_template, and fslvbm_3_proc.

## Requirements

Requires the installation of ANTS (for Registration and Atropos) and FSL (for Randomise).

## Usage

Add later

## Results

Abstract to be submitted to AAN/ACTRIMS 2022.
