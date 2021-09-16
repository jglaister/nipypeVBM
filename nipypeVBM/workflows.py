import os, glob  # system functions

import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util

from .interfaces import GenerateTemplate


def create_nipypevbm_workflow(output_root, sigma):
    wf = pe.Workflow(name='nipypevbm', base_dir=output_root)
    wf_root = os.path.join(output_root, 'nipypevbm')

    input_node = pe.Node(
        interface=util.IdentityInterface(
            fields=['struct_files', 'GM_template', 'design_mat', 'tcon']),
        name='input_node')

    bet_workflow = create_bet_workflow(wf_root)
    wf.connect(input_node, 'struct_files', bet_workflow, 'input_node.struct_files')

    preproc_workflow = create_preproc_workflow(wf_root)
    wf.connect(bet_workflow, 'output_node.brain_files', preproc_workflow, 'input_node.brain_files')
    wf.connect(input_node, 'GM_template', preproc_workflow, 'input_node.GM_template')

    proc_workflow = create_proc_workflow(wf_root, sigma)
    wf.connect(preproc_workflow, 'output_node.GM_files', proc_workflow, 'input_node.GM_files')
    wf.connect(preproc_workflow, 'output_node.GM_template', proc_workflow, 'input_node.GM_template')
    wf.connect(input_node, 'design_mat', proc_workflow, 'input_node.design_mat')
    wf.connect(input_node, 'tcon', proc_workflow, 'input_node.tcon')

    # TODO: Add output node and move files

    return wf


def create_bet_workflow(output_root):
    # Set up workflow
    wf = pe.Workflow(name='fslvbm_1_bet', base_dir=output_root)

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


def create_preproc_workflow(output_root, gm_alg='atropos'):
    wf = pe.Workflow(name='fslvbm_2_template', base_dir=output_root)
    wf_root = os.path.join(output_root, 'fslvbm_2_template')

    input_node = pe.Node(
        interface=util.IdentityInterface(fields=['brain_files', 'GM_template']),
        name='input_node')

    if gm_alg is 'fslfast':
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
        
    elif gm_alg is 'atropos':
        #Register template to brain
        deformable_priors = pe.MapNode(ants.Registration(), iterfield=['fixed_image'], name='deformable_priors')
        deformable_priors.inputs.dimension = 3
        deformable_priors.inputs.interpolation = 'Linear'
        deformable_priors.inputs.metric = ['MI', 'MI', 'MI']
        deformable_priors.inputs.metric_weight = [1.0, 1.0, 1.0]
        deformable_priors.inputs.radius_or_number_of_bins = [32, 32, 32]
        deformable_priors.inputs.sampling_strategy = ['Regular', 'Regular', 'Regular']
        deformable_priors.inputs.sampling_percentage = [0.25, 0.25, 0.25]
        deformable_priors.inputs.transforms = ['Rigid', 'Affine', 'SyN']
        deformable_priors.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3, 0)]
        deformable_priors.inputs.number_of_iterations = [[100, 50, 25], [100, 50, 25], [100, 10, 5]]
        deformable_priors.inputs.convergence_threshold = [1e-6, 1e-6, 1e-4]
        deformable_priors.inputs.convergence_window_size = [10, 10, 10]
        deformable_priors.inputs.smoothing_sigmas = [[4, 2, 1], [4, 2, 1], [2, 1, 0]]
        deformable_priors.inputs.sigma_units = ['vox', 'vox', 'vox']
        deformable_priors.inputs.shrink_factors = [[4, 2, 1], [4, 2, 1], [4, 2, 1]]
        deformable_priors.inputs.write_composite_transform = True
        deformable_priors.inputs.initial_moving_transform_com = 1
        deformable_priors.inputs.output_warped_image = True
        #Template file
        deformable_priors.inputs.moving_image = '/home/j/jiwonoh/jglaist1/atlas/Oasis/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellum_rai.nii.gz'
        wf.connect(input_node, 'brain_files', deformable_priors, 'moving_image')

        #Warp priors
        warp_priors = pe.MapNode(ants.ApplyTransforms(), iterfield=['reference_image', 'transforms'], name='warp_priors')
        warp_priors.inputs.input_image = '/home/j/jiwonoh/jglaist1/atlas/Oasis/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_glm_6labelsJointFusion_rai.nii.gz'
        wf.connect(input_node, 'brain_files', warp_priors, 'reference_image')
        wf.connect(deformable_priors, 'composite_transform', warp_priors, 'transforms')

        ants_atropos = pe.MapNode(ants.Atropos(), iterfield=['intensity_images', 'prior_image'], name='ants_atropos')
        ants_atropos.inputs.dimension = 3
        ants_atropos.inputs.initialization = 'PriorLabelImage'
        ants_atropos.inputs.number_of_tissue_classes = 6
        ants_atropos.inputs.save_posteriors = True
        wf.connect(input_node, 'brain_files', ants_atropos, 'intensity_images')
        #Connect warped prior
        wf.connect(input_node, 'brain_files', ants_atropos, 'prior_image')
    else:
        print('GM segmentation must be fslfast or atropos')


    # Affine registration of GM from FAST to GM template
    affine_reg_to_gm = pe.MapNode(ants.Registration(), iterfield=['moving_image'], name='affine_reg_to_GM')
    affine_reg_to_gm.inputs.dimension = 3
    affine_reg_to_gm.inputs.interpolation = 'Linear'
    affine_reg_to_gm.inputs.metric = ['MI', 'MI']
    affine_reg_to_gm.inputs.metric_weight = [1.0, 1.0]
    affine_reg_to_gm.inputs.radius_or_number_of_bins = [32, 32]
    affine_reg_to_gm.inputs.sampling_strategy = ['Regular', 'Regular']
    affine_reg_to_gm.inputs.sampling_percentage = [0.25, 0.25]
    affine_reg_to_gm.inputs.transforms = ['Rigid', 'Affine']
    affine_reg_to_gm.inputs.transform_parameters = [(0.1,), (0.1,)]
    affine_reg_to_gm.inputs.number_of_iterations = [[100, 50, 25], [100, 50, 25]]
    affine_reg_to_gm.inputs.convergence_threshold = [1e-6, 1e-6]
    affine_reg_to_gm.inputs.convergence_window_size = [10, 10]
    affine_reg_to_gm.inputs.smoothing_sigmas = [[4, 2, 1], [4, 2, 1]]
    affine_reg_to_gm.inputs.sigma_units = ['vox', 'vox']
    affine_reg_to_gm.inputs.shrink_factors = [[4, 2, 1], [4, 2, 1]]
    affine_reg_to_gm.inputs.write_composite_transform = True
    affine_reg_to_gm.inputs.initial_moving_transform_com = 1
    affine_reg_to_gm.inputs.output_warped_image = True
    wf.connect(split_priors, 'out2', affine_reg_to_gm, 'moving_image')
    wf.connect(input_node, 'GM_template', affine_reg_to_gm, 'fixed_image')
    #affine_reg_to_gm = pe.MapNode(interface=fsl.FLIRT(),
    #                              iterfield=['in_file'],
    #                              name='affine_reg_to_GM')
    #wf.connect(split_priors, 'out2', affine_reg_to_gm, 'in_file')
    #wf.connect(input_node, 'GM_template', affine_reg_to_gm, 'reference')

    # Average 4D template and its flipped template to create an initial template
    affine_4d_template = pe.Node(interface=fsl.Merge(),
                                 name='affine_4D_template')
    affine_4d_template.inputs.dimension = 't'
    wf.connect(affine_reg_to_gm, 'warped_image', affine_4d_template, 'in_files')
    #wf.connect(affine_reg_to_gm, 'out_file', affine_4d_template, 'in_files')

    affine_template = pe.Node(interface=GenerateTemplate(),
                              name='affine_template')
    wf.connect(affine_4d_template, 'merged_file', affine_template, 'input_file')

    # Nonlinear registration to initial template
    nonlinear_reg_to_temp = pe.MapNode(interface=ants.Registration(),
                                       iterfield=['moving_image'],
                                       name='nonlinear_reg_to_temp')
    nonlinear_reg_to_temp.inputs.dimension = 3
    nonlinear_reg_to_temp.inputs.interpolation = 'Linear'
    nonlinear_reg_to_temp.inputs.metric = ['MI', 'MI', 'MI']
    nonlinear_reg_to_temp.inputs.metric_weight = [1.0, 1.0, 1.0]
    nonlinear_reg_to_temp.inputs.radius_or_number_of_bins = [32, 32, 32]
    nonlinear_reg_to_temp.inputs.sampling_strategy = ['Regular', 'Regular', 'Regular']
    nonlinear_reg_to_temp.inputs.sampling_percentage = [0.25, 0.25, 0.25]
    nonlinear_reg_to_temp.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    nonlinear_reg_to_temp.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3, 0)]
    nonlinear_reg_to_temp.inputs.number_of_iterations = [[100, 50, 25], [100, 50, 25], [100, 10, 5]]
    nonlinear_reg_to_temp.inputs.convergence_threshold = [1e-6, 1e-6, 1e-4]
    nonlinear_reg_to_temp.inputs.convergence_window_size = [10, 10, 10]
    nonlinear_reg_to_temp.inputs.smoothing_sigmas = [[4, 2, 1], [4, 2, 1], [2, 1, 0]]
    nonlinear_reg_to_temp.inputs.sigma_units = ['vox', 'vox', 'vox']
    nonlinear_reg_to_temp.inputs.shrink_factors = [[4, 2, 1], [4, 2, 1], [4, 2, 1]]
    nonlinear_reg_to_temp.inputs.write_composite_transform = True
    nonlinear_reg_to_temp.inputs.initial_moving_transform_com = 1
    nonlinear_reg_to_temp.inputs.output_warped_image = True
    wf.connect(split_priors, 'out2', nonlinear_reg_to_temp, 'moving_image')
    wf.connect(affine_template, 'template_file', nonlinear_reg_to_temp, 'fixed_image')
    #nonlinear_reg_to_temp = pe.MapNode(interface=fsl.FNIRT(),
    #                                   iterfield=['in_file', 'affine_file'],
    #                                   name='nonlinear_reg_to_temp')
    ## Check for config file in FSLDIR, otherwise uses defaults
    ##config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf', 'GM_2_MNI152GM_2mm.cnf')
    ##if os.path.exists(config_file):
    ##    nonlinear_reg_to_temp.inputs.config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf',
    ##                                                            'GM_2_MNI152GM_2mm.cnf')
    #wf.connect(split_priors, 'out2', nonlinear_reg_to_temp, 'in_file')
    #wf.connect(affine_template, 'template_file', nonlinear_reg_to_temp, 'ref_file')
    #wf.connect(affine_reg_to_gm, 'out_matrix_file', nonlinear_reg_to_temp, 'affine_file')

    nonlinear_4d_template = pe.Node(interface=fsl.Merge(),
                                    name='nonlinear_4d_template')
    nonlinear_4d_template.inputs.dimension = 't'
    wf.connect(nonlinear_reg_to_temp, 'warped_image', nonlinear_4d_template, 'in_files')
    #wf.connect(nonlinear_reg_to_temp, 'warped_file', nonlinear_4d_template, 'in_files')

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


def create_proc_workflow(output_root, sigma=2):
    wf = pe.Workflow(name='fslvbm_3_proc', base_dir=output_root)

    input_node = pe.Node(
        interface=util.IdentityInterface(fields=['GM_files', 'GM_template', 'design_mat', 'tcon']),
        name='input_node')

    nonlinear_reg_to_temp = pe.MapNode(interface=ants.Registration(),
                                       iterfield=['moving_image'],
                                       name='nonlinear_reg_to_temp')
    nonlinear_reg_to_temp.inputs.dimension = 3
    nonlinear_reg_to_temp.inputs.interpolation = 'Linear'
    nonlinear_reg_to_temp.inputs.metric = ['MI', 'MI', 'MI']
    nonlinear_reg_to_temp.inputs.metric_weight = [1.0, 1.0, 1.0]
    nonlinear_reg_to_temp.inputs.radius_or_number_of_bins = [32, 32, 32]
    nonlinear_reg_to_temp.inputs.sampling_strategy = ['Regular', 'Regular', 'Regular']
    nonlinear_reg_to_temp.inputs.sampling_percentage = [0.25, 0.25, 0.25]
    nonlinear_reg_to_temp.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    nonlinear_reg_to_temp.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3, 0)]
    nonlinear_reg_to_temp.inputs.number_of_iterations = [[100, 50, 25], [100, 50, 25], [100, 10, 5]]
    nonlinear_reg_to_temp.inputs.convergence_threshold = [1e-6, 1e-6, 1e-4]
    nonlinear_reg_to_temp.inputs.convergence_window_size = [10, 10, 10]
    nonlinear_reg_to_temp.inputs.smoothing_sigmas = [[4, 2, 1], [4, 2, 1], [2, 1, 0]]
    nonlinear_reg_to_temp.inputs.sigma_units = ['vox', 'vox', 'vox']
    nonlinear_reg_to_temp.inputs.shrink_factors = [[4, 2, 1], [4, 2, 1], [4, 2, 1]]
    nonlinear_reg_to_temp.inputs.write_composite_transform = False
    nonlinear_reg_to_temp.inputs.initial_moving_transform_com = 1
    nonlinear_reg_to_temp.inputs.output_warped_image = True
    wf.connect(input_node, 'GM_files', nonlinear_reg_to_temp, 'moving_image')
    wf.connect(input_node, 'GM_template', nonlinear_reg_to_temp, 'fixed_image')
    #nonlinear_reg_to_temp = pe.MapNode(interface=fsl.FNIRT(),
    #                                   iterfield=['in_file'],
    #                                   name='nonlinear_reg_to_temp')
    ## Use defaults for now
    #config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf', 'GM_2_MNI152GM_2mm.cnf')
    #if os.path.exists(config_file):
    #    nonlinear_reg_to_temp.inputs.config_file = os.path.join(os.environ['FSLDIR'], 'src', 'fnirt', 'fnirtcnf',
    #                                                            'GM_2_MNI152GM_2mm.cnf')
    #nonlinear_reg_to_temp.inputs.jacobian_file = True
    #wf.connect(input_node, 'GM_files', nonlinear_reg_to_temp, 'in_file')
    #wf.connect(input_node, 'GM_template', nonlinear_reg_to_temp, 'ref_file')

    split_transforms = pe.MapNode(interface=util.Split(),
                              iterfield=['inlist'],
                              name='split_transforms')
    split_transforms.inputs.splits = [1, 1]
    split_transforms.inputs.squeeze = True
    wf.connect(nonlinear_reg_to_temp, 'forward_transforms', split_transforms, 'inlist')

    create_jac = pe.MapNode(interface=ants.utils.CreateJacobianDeterminantImage(),
                            iterfield=['deformationField'],
                            name='create_jac')
    create_jac.inputs.imageDimension = 3
    create_jac.inputs.outputImage = 'Jacobian.nii.gz'
    create_jac.inputs.doLogJacobian = 0
    create_jac.inputs.useGeometric = 1
    wf.connect(split_transforms, 'out2', create_jac, 'deformationField')

    # Multiply JAC and GM
    gm_mul_jac = pe.MapNode(interface=fsl.ImageMaths(),
                            iterfield=['in_file', 'in_file2'],
                            name='gm_mul_jac')
    gm_mul_jac.inputs.op_string = '-mul'
    wf.connect(nonlinear_reg_to_temp, 'warped_image', gm_mul_jac, 'in_file')
    #wf.connect(nonlinear_reg_to_temp, 'warped_file', gm_mul_jac, 'in_file')
    wf.connect(create_jac, 'jacobian_image', gm_mul_jac, 'in_file2')
    #wf.connect(nonlinear_reg_to_temp, 'jacobian_file', gm_mul_jac, 'in_file2')

    # Merge GMs
    gm_merge = pe.Node(interface=fsl.Merge(),
                                    name='gm_merge')
    gm_merge.inputs.dimension = 't'
    wf.connect(nonlinear_reg_to_temp, 'warped_image', gm_merge, 'in_files')
    #wf.connect(nonlinear_reg_to_temp, 'warped_file', gm_merge, 'in_files')

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
    final_randomise = pe.Node(interface=fsl.model.Randomise(), name='final_randomise')
    final_randomise.inputs.base_name = 'GM_mod_merg_s' + str(sigma)
    final_randomise.inputs.tfce = True
    final_randomise.inputs.num_perm = 1000
    wf.connect(gaussian_filter, 'out_file', final_randomise, 'in_file')
    wf.connect(gm_mask, 'out_file', final_randomise, 'mask')
    wf.connect(input_node, 'design_mat', final_randomise, 'design_mat')
    wf.connect(input_node, 'tcon', final_randomise, 'tcon')

    return wf



