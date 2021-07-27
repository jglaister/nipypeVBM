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
            fields=['struct_files', 'GM_template', 'designmat', 'designcon']),
        name='input_node')

    bet_workflow = create_bet_workflow(wf_root)
    wf.connect(input_node, 'struct_files', bet_workflow.inputs.input_node.struct_files, 'struct_files')

    preproc_workflow = create_preproc_workflow(wf_root)
    wf.connect(bet_workflow.output_node, 'brain_files', bet_workflow.inputs.input_node.struct_files, 'brain_files')

def create_bet_workflow(output_root):
    # Set up workflow
    wf = pe.Workflow(name='fslvbm_1_bet',base_dir=os.path.join(output_root, 'fslvbm_1_bet'))

    # Find scans
    #input_path = os.path.join(scan_directory, '*' + scan_suffix + '*')
    #input_files = sorted(glob.glob(input_path))

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
        interface=util.IdentityInterface(fields=['brain_files','GM_template']),
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

    #Affine registration of GM from FAST to GM template
    affine_reg_to_GM = pe.MapNode(interface=fsl.FLIRT(),
                                  iterfield=['in_file'],
                                  name='affine_reg_to_GM')
    #Use defaults for now
    wf.connect(split_priors, 'out2', affine_reg_to_GM, 'in_file')
    wf.connect(input_node, 'GM_template', affine_reg_to_GM, 'reference')

    affine_4d_template = pe.Node(interface=fsl.Merge(),
                                 name='affine_4D_template')
    affine_4d_template.inputs.dimension = 't'
    wf.connect(affine_reg_to_GM, 'out_file', affine_4d_template, 'in_files')

    affine_template = pe.Node(interface=GenerateTemplate(),
                              name='affine_template')
    wf.connect(affine_4d_template, 'merged_file', affine_template, 'input_file')


    #Nonlinear registration to initial template
    nonlinear_reg_to_temp = pe.MapNode(interface=fsl.FNIRT(),
                                       iterfield=['in_file'],
                                       name='nonlinear_reg_to_temp')
    # Use defaults for now
    #nonlinear_reg_to_temp.inputs.warped_file = 'test.nii.gz'
    wf.connect(split_priors, 'out3', nonlinear_reg_to_temp, 'in_file')
    wf.connect(affine_template, 'template_file', nonlinear_reg_to_temp, 'ref_file')

    nonlinear_4d_template = pe.Node(interface=fsl.Merge(),
                                    name='nonlinear_4d_template')
    nonlinear_4d_template.inputs.dimension = 't'
    wf.connect(nonlinear_reg_to_temp, 'warped_file ', nonlinear_4d_template, 'in_files')
    '''
    nonlinear_template = pe.MapNode(interface=GenerateTemplate(),
                                    name='nonlinear_template')
    wf.connect(nonlinear_4d_template, 'merged_file', nonlinear_template, 'in_file')

    output_node = pe.Node(
        interface=util.IdentityInterface(fields=['template_file', 'GM_files']),
        name='output_node')
    wf.connect(nonlinear_4d_template, 'template_file', output_node, 'template_file')
    wf.connect(split_priors, 'out3', output_node, 'out_files')
    '''

    return wf
    #fsl_reg $OUTPUT / bet /${SUBID}_GM $GPRIORS $OUTPUT / bet /${SUBID}_GM_to_T - a


def create_proc_workflow(output_root):
    wf = pe.Workflow(name='fslvbm_3_proc', base_dir=os.path.join(output_root,'fslvbm_3_proc'))

    input_node = pe.Node(
        interface=util.IdentityInterface(fields=['GM_files', 'GM_template', 'designmat', 'designcon']),
        name='input_node')

    nonlinear_reg_to_temp = pe.MapNode(interface=fsl.FNIRT(),
                                      iterfield=['in_file'],
                                      name='nonlinear_reg_to_temp')
    # Use defaults for now
    #Config file?
    wf.connect(input_node, 'GM_files', nonlinear_reg_to_temp, 'in_file')
    wf.connect(input_node, 'GM_template', nonlinear_reg_to_temp, 'reference')

    #Multiply JAC and GM
    gm_mul_jac = pe.MapNode(interface=fsl.ImageMaths(),
                            iterfield=['in_file','in_file2'],
                            name='gm_mul_jac')
    gm_mul_jac.inputs.op_string = '-mul'
    wf.connect(nonlinear_reg_to_temp, 'out_file', gm_mul_jac, 'in_file')
    wf.connect(input_node, 'GM_files', gm_mul_jac, 'in_file2')

    #Merge GMs
    #fslmerge - t GM_merg     \`\${FSLDIR} / bin / imglob.. / struc / * _GM_to_template_GM. *\`
    #fslmerge - t GM_mod_merg \`\${FSLDIR} / bin / imglob.. / struc / * _GM_to_template_GM_mod. *\`
    #fslmaths GM_merg - Tmean - thr 0.01 - bin GM_mask - odt char

    #Multiply by Gaussian for s = 2, 3, 4

    #Randomise



