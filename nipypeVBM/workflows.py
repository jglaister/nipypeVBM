import os, glob  # system functions

import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util

from .interfaces import GenerateTemplate


def create_nipypevbm_workflow(output_root):
    wf_root = os.path.join(output_root, 'nipypevbm')
    wf = pe.Workflow(name='nipypevbm', base_dir=wf_root)

    input_node = pe.Node(
        interface=util.IdentityInterface(
            fields=['struct_files', 'GM_template', 'design_mat', 'tcon']),
        name='input_node')

    bet_workflow = create_bet_workflow(wf_root)
    wf.connect(input_node, 'struct_files', bet_workflow.inputs.input_node, 'struct_files')

    preproc_workflow = create_preproc_workflow(wf_root)
    wf.connect(bet_workflow.output_node, 'brain_files', preproc_workflow.inputs.input_node, 'brain_files')
    wf.connect(input_node, 'GM_template', preproc_workflow.inputs.input_node, 'GM_template')

    proc_workflow = create_proc_workflow(wf_root)
    wf.connect(preproc_workflow.output_node, 'GM_files', preproc_workflow.inputs.input_node, 'GM_files')
    wf.connect(preproc_workflow.output_node, 'GM_template', preproc_workflow.inputs.input_node, 'GM_template')
    wf.connect(input_node, 'design_mat', preproc_workflow.inputs.input_node, 'design_mat')
    wf.connect(input_node, 'tcon', preproc_workflow.inputs.input_node, 'tcon')

    # TODO: Add output node and move files

    return wf


def create_bet_workflow(output_root):
    # Set up workflow
    wf = pe.Workflow(name='fslvbm_1_bet',base_dir=os.path.join(output_root, 'fslvbm_1_bet'))

    input_node = pe.Node(
        interface=util.IdentityInterface(fields=['struct_files']),
        name='input_node')
    
    fsl_bet = pe.MapNode(interface=fsl.BET(),
                         iterfield=['in_file'],
                         name='fsl_bet')
    fsl_bet.inputs.frac = 0.4
    fsl_bet.inputs.mask = True
    wf.connect(input_node, 'struct_files', fsl_bet, 'in_file')

    # Set up output node with brain mask and masked brains
    output_node = pe.Node(
        interface=util.IdentityInterface(fields=['mask_files', 'brain_files']),
        name='output_node')
    wf.connect(fsl_bet, 'mask_file', output_node, 'mask_files')
    wf.connect(fsl_bet, 'out_file', output_node, 'brain_files')

    return wf


def create_preproc_workflow(output_root):
    wf = pe.Workflow(name='fslvbm_2_template', base_dir=os.path.join(output_root,'fslvbm_2_template'))

    input_node = pe.Node(
        interface=util.IdentityInterface(fields=['brain_files', 'GM_template']),
        name='input_node')

    fsl_fast = pe.MapNode(interface=fsl.FAST(),
                          iterfield=['in_files'],
                          name='fsl_fast')
    fsl_fast.inputs.mixel_smooth = 0.3
    fsl_fast.inputs.hyper = 0.1
    wf.connect(input_node, 'brain_files', fsl_fast, 'in_files')

    split_priors = pe.MapNode(interface=util.Split(),
                              iterfield=['inlist'],
                              name='split_priors')
    split_priors.inputs.splits = [1, 1, 1]
    split_priors.inputs.squeeze = True
    wf.connect(fsl_fast, 'partial_volume_files', split_priors, 'inlist')

    # Affine registration of GM from FAST to GM template
    affine_reg_to_gm = pe.MapNode(interface=fsl.FLIRT(),
                                  iterfield=['in_file'],
                                  name='affine_reg_to_GM')
    wf.connect(split_priors, 'out2', affine_reg_to_gm, 'in_file')
    wf.connect(input_node, 'GM_template', affine_reg_to_gm, 'reference')

    # Average 4D template and its flipped template to create an initial template
    affine_4d_template = pe.Node(interface=fsl.Merge(),
                                 name='affine_4D_template')
    affine_4d_template.inputs.dimension = 't'
    wf.connect(affine_reg_to_gm, 'out_file', affine_4d_template, 'in_files')

    affine_template = pe.Node(interface=GenerateTemplate(),
                              name='affine_template')
    wf.connect(affine_4d_template, 'merged_file', affine_template, 'input_file')

    # Nonlinear registration to initial template
    nonlinear_reg_to_temp = pe.MapNode(interface=fsl.FNIRT(),
                                       iterfield=['in_file'],
                                       name='nonlinear_reg_to_temp')
    # Check for config file in FSLDIR, otherwise uses defaults
    config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf', 'GM_2_MNI152GM_2mm.cnf')
    if os.path.exists(config_file):
        nonlinear_reg_to_temp.inputs.config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf',
                                                                'GM_2_MNI152GM_2mm.cnf')
    wf.connect(split_priors, 'out2', nonlinear_reg_to_temp, 'in_file')
    wf.connect(affine_template, 'template_file', nonlinear_reg_to_temp, 'ref_file')

    nonlinear_4d_template = pe.Node(interface=fsl.Merge(),
                                    name='nonlinear_4d_template')
    nonlinear_4d_template.inputs.dimension = 't'
    wf.connect(nonlinear_reg_to_temp, 'warped_file', nonlinear_4d_template, 'in_files')

    # TODO: Allow for variable size cohorts instead of matched sizes
    nonlinear_template = pe.Node(interface=GenerateTemplate(),
                                 name='nonlinear_template')
    wf.connect(nonlinear_4d_template, 'merged_file', nonlinear_template, 'input_file')

    output_node = pe.Node(
        interface=util.IdentityInterface(fields=['GM_template', 'GM_files']),
        name='output_node')
    wf.connect(nonlinear_template, 'template_file', output_node, 'GM_template')
    wf.connect(split_priors, 'out2', output_node, 'GM_files')

    return wf


def create_proc_workflow(output_root, sigma=3):
    wf = pe.Workflow(name='fslvbm_3_proc', base_dir=os.path.join(output_root,'fslvbm_3_proc'))

    input_node = pe.Node(
        interface=util.IdentityInterface(fields=['GM_files', 'GM_template', 'design_mat', 'tcon']),
        name='input_node')

    nonlinear_reg_to_temp = pe.MapNode(interface=fsl.FNIRT(),
                                       iterfield=['in_file'],
                                       name='nonlinear_reg_to_temp')
    # Use defaults for now
    config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf', 'GM_2_MNI152GM_2mm.cnf')
    if os.path.exists(config_file):
        nonlinear_reg_to_temp.inputs.config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf',
                                                                'GM_2_MNI152GM_2mm.cnf')
    nonlinear_reg_to_temp.inputs.jacobian_file = True
    wf.connect(input_node, 'GM_files', nonlinear_reg_to_temp, 'in_file')
    wf.connect(input_node, 'GM_template', nonlinear_reg_to_temp, 'ref_file')

    # Multiply JAC and GM
    gm_mul_jac = pe.MapNode(interface=fsl.ImageMaths(),
                            iterfield=['in_file', 'in_file2'],
                            name='gm_mul_jac')
    gm_mul_jac.inputs.op_string = '-mul'
    wf.connect(nonlinear_reg_to_temp, 'warped_file', gm_mul_jac, 'in_file')
    wf.connect(nonlinear_reg_to_temp, 'jacobian_file', gm_mul_jac, 'in_file2')

    # Merge GMs
    gm_merge = pe.Node(interface=fsl.Merge(),
                                    name='gm_merge')
    gm_merge.inputs.dimension = 't'
    wf.connect(nonlinear_reg_to_temp, 'warped_file', gm_merge, 'in_files')

    gm_mod_merge = pe.Node(interface=fsl.Merge(),
                       name='gm_mod_merge')
    gm_mod_merge.inputs.dimension = 't'
    wf.connect(gm_mul_jac, 'out_file', gm_mod_merge, 'in_files')

    gm_mask = pe.Node(interface=fsl.ImageMaths(), name='gm_mask')
    gm_mask.inputs.op_string = '-Tmean -thr 0.01 -bin'
    gm_mask.inputs.out_data_type = 'char'
    wf.connect(gm_merge, 'merged_file', gm_mask, 'in_file')

    gaussian_filter = pe.Node(interface=fsl.ImageMaths(), name='gaussian')
    gaussian_filter.inputs.op_string = '-s ' + str(sigma)
    wf.connect(gm_mod_merge, 'merged_file', gaussian_filter, 'in_file')

    init_randomise = pe.Node(interface=fsl.model.Randomise(), name='randomise')
    init_randomise.inputs.base_name = 'GM_mod_merg_s' + str(sigma)

    wf.connect(gaussian_filter, 'out_file', init_randomise, 'in_file')
    wf.connect(gm_mask, 'out_file', init_randomise, 'mask')
    wf.connect(input_node, 'design_mat', init_randomise, 'design_mat')
    wf.connect(input_node, 'tcon', init_randomise, 'tcon')

    # TODO: Add output node

    return wf



