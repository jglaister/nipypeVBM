import os.path
import glob

import nipype.interfaces.base as base
import nipype.utils.filemanip as fip


class GenerateTemplateInputSpec(base.BaseInterfaceInputSpec):
    input_file = base.File(exists=True, desc='input image', mandatory=True)
    flip_axis = base.traits.Int(0, desc='Axis number to flip (-1 to not flip)', usedefault=True)
    output_name = base.traits.Str(desc='Filename for output template')

class GenerateTemplateOutputSpec(base.TraitedSpec):
    template_file = base.File(exists=True, desc='output template')


class GenerateTemplate(base.BaseInterface):
    input_spec = GenerateTemplateInputSpec
    output_spec = GenerateTemplateOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        vol_obj = nib.load(self.inputs.input_file)
        vol_data = vol_obj.get_fdata()

        if self.inputs.flip_axis == -1:
            template_data = np.average(vol_data, axis=-1)
        else:
            template_data = (np.average(vol_data, axis=-1) + np.average(np.flip(vol_data, axis=self.inputs.flip_axis), axis=-1)) / 2

        template_obj = nib.Nifti1Image(template_data, vol_obj.affine, vol_obj.header)
        if base.isdefined(self.inputs.output_name):
            output_name = self.inputs.output_name + '.nii.gz'
        else:
            output_name = 'template.nii.gz'
        template_obj.to_filename(output_name)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        if base.isdefined(self.inputs.output_name):
            outputs['template_file'] = os.path.abspath(self.inputs.output_name + '.nii.gz')
        else:
            outputs['template_file'] = os.path.abspath('template.nii.gz')
        return outputs


class GeneratePriorsInputSpec(base.BaseInterfaceInputSpec):
    reference_file = base.File(exists=True, desc='input image', mandatory=True)
    prior_4D_file = base.File(exists=True, desc='input image', mandatory=True)


class GeneratePriorsOutputSpec(base.TraitedSpec):
    prior_3D_files = base.File(exists=True, desc='output template')
    prior_string = base.traits.String()


class GeneratePriors(base.BaseInterface):
    input_spec = GeneratePriorsInputSpec
    output_spec = GeneratePriorsOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        vol_obj = nib.load(self.inputs.prior_4D_file)
        vol_data = vol_obj.get_fdata()
        ref_obj = nib.load(self.inputs.reference_file)

        bg_priors = 1 - np.sum(vol_data, 3)
        output_filename = fip.split_filename(self.inputs.reference_file)[1] + '_prior01.nii.gz'
        prior_obj = nib.Nifti1Image(bg_priors, ref_obj.affine, ref_obj.header)
        prior_obj.to_filename(output_filename)

        for i in range(vol_data.shape[3]):
            output_filename = fip.split_filename(self.inputs.reference_file)[1] + '_prior0' + str(i + 2) + '.nii.gz'
            img = vol_data[:, :, :, i]
            prior_obj = nib.Nifti1Image(img, ref_obj.affine, ref_obj.header)
            prior_obj.to_filename(output_filename)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        basename = fip.split_filename(self.inputs.reference_file)[1]
        outputs['prior_3D_files'] = sorted(glob(os.path.abspath(basename + '_prior*.nii.gz')))
        outputs['prior_string'] = os.path.abspath(basename + '_prior%02d.nii.gz')
        return outputs


