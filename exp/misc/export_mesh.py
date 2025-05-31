from lib.kits.basic import *

from lib.body_models.common import make_SKEL
from lib.utils.data import to_tensor
from lib.info.log import get_logger
import trimesh
import tyro

def export_mesh(
    input_path   : Path = PM.outputs / 'demos' / 'HSMR-ballerina.png.npz',
    outputs_root : Path = PM.outputs / 'demos' / 'HSMR-ballerina',
    device       : str  = 'cuda:0',
) -> None :
    input = np.load(input_path, allow_pickle=True)
    poses = to_tensor(input['poses'], device=device)  # (B, 46)
    betas = to_tensor(input['betas'], device=device)  # (B, 10)

    if len(poses) == 0 or len(betas) == 0:
        get_logger(brief=True).error(f'🚫 No poses or betas found in {input_path}.')
        return
    skel_model = make_SKEL(device=device)
    skel_output = skel_model(
        betas    = betas,
        poses    = poses,
        skelmesh = True,  # enable this if you want the skeleton mesh
    )
    for i in range(len(poses)):
        # Export SMPL-style skin mesh.
        skin_verts = skel_output.skin_verts  # (B, Vi, 3)
        skin_faces = skel_model.skin_f  # (F, 3)
        skin_mesh = trimesh.Trimesh(
                vertices=skin_verts[i].detach().cpu().numpy(),
                faces=skin_faces.cpu().numpy(),
            )
        skin_mesh_fn = input_path.with_suffix(f'.skin_{i}.obj')
        skin_mesh.export(skin_mesh_fn, file_type='obj')

        # Export BSM-style skeleton mesh.
        skel_verts = skel_output.skel_verts  # (B, Ve, 3)
        skel_faces = skel_model.skel_f  # (F, 3)
        skel_mesh = trimesh.Trimesh(
                vertices = skel_verts[i].detach().cpu().numpy(),
                faces    = skel_faces.cpu().numpy(),
            )
        skel_mesh_fn = input_path.with_suffix(f'.skel_{i}.obj')
        skel_mesh.export(skel_mesh_fn, file_type='obj')

        get_logger(brief=True).info(f'🎨 Rendering results are under {outputs_root}.')


if __name__ == '__main__':
    import tyro
    tyro.cli(export_mesh)