import os.path

import nipype.interfaces.base as base
import nipype.utils.filemanip as fip


class GenerateTemplateInputSpec(base.BaseInterfaceInputSpec):
    input_file = base.File(exists=True, desc='input image', mandatory=True)
    flip_axis = base.traits.Int(desc='Axis number to flip (-1 to not flip)', default_value=1)
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
            template_data = (np.average(vol_data, axis=-1) + np.average(np.flip(vol_data, axis=1), axis=-1)) / 2

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



