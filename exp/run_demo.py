from lib.kits.hsmr_demo import *

def main():
    # ⛩️ 0. Preparation.
    args = parse_args()
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    monitor = TimeMonitor()

    # ⛩️ 1. Preprocess.

    with monitor('Data Preprocessing'):
        with monitor('Load Inputs'):
            raw_imgs, inputs_meta = load_inputs(args)

        with monitor('Detector Initialization'):
            get_logger(brief=True).info('🧱 Building detector.')
            detector = build_detector(
                    batch_size   = args.det_bs,
                    max_img_size = args.det_mis,
                    device       = args.device,
                )

        with monitor('Detecting'):
            get_logger(brief=True).info(f'🖼️ Detecting...')
            detector_outputs = detector(raw_imgs)

        with monitor('Patching & Loading'):
            patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, args.max_instances)  # N * (256, 256, 3)
        if len(patches) == 0:
            get_logger(brief=True).error(f'🚫 No human instance detected. Please ensure the validity of your inputs!')
        get_logger(brief=True).info(f'🔍 Totally {len(patches)} human instances are detected.')


    # ⛩️ 2. Human skeleton and mesh recovery.
    with monitor('Pipeline Initialization'):
        get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)

    with monitor('Recovery'):
        get_logger(brief=True).info(f'🏃 Recovering with B={args.rec_bs}...')
        pd_params, pd_cam_t = [], []
        for bw in asb(total=len(patches), bs_scope=args.rec_bs, enable_tqdm=True):
            patches_i = np.concatenate(patches[bw.sid:bw.eid], axis=0)  # (N, 256, 256, 3)
            patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255  # (N, 256, 256, 3)
            patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)  # (N, 3, 256, 256)
            with torch.no_grad():
                outputs = pipeline(patches_normalized_i)
            pd_params.append({k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()})
            pd_cam_t.append(outputs['pd_cam_t'].detach().cpu().clone())

        pd_params = assemble_dict(pd_params, expand_dim=False)  # [{k:[x]}, {k:[y]}] -> {k:[x, y]}
        pd_cam_t = torch.cat(pd_cam_t, dim=0)
        dump_results = {
                'patch_cam_t' : pd_cam_t.numpy(),
                **{k: v.numpy() for k, v in pd_params.items()},
            }

        get_logger(brief=True).info(f'🤌 Preparing meshes...')
        m_skin, m_skel = prepare_mesh(pipeline, pd_params)
        get_logger(brief=True).info(f'🏁 Done.')


    # ⛩️ 3. Postprocess.
    with monitor('Visualization'):
        if args.ignore_skel:
            m_skel = None
        results, full_cam_t = visualize_full_img(pd_cam_t, raw_imgs, det_meta, m_skin, m_skel, args.have_caption)
        dump_results['full_cam_t'] = full_cam_t
        # Save rendering and dump results.
        if inputs_meta['type'] == 'video':
            seq_name = f'{pipeline.name}-' + inputs_meta['seq_name']
            save_video(results, outputs_root / f'{seq_name}.mp4')
            np.savez(outputs_root / f'{seq_name}.npz', **dump_results)
        elif inputs_meta['type'] == 'imgs':
            img_names = [f'{pipeline.name}-{fn.name}' for fn in inputs_meta['img_fns']]
            # Dump data for each image separately, here `i` refers to images, `j` refers to image patches.
            cur_patch_j = 0
            for i, img_name in enumerate(tqdm(img_names, desc='Saving images')):
                dump_results_i = {k: v[cur_patch_j:cur_patch_j+det_meta['n_patch_per_img'][i]] for k, v in dump_results.items()}
                dump_results_i['bbx_cs'] = det_meta['bbx_cs'][i]
                cur_patch_j += det_meta['n_patch_per_img'][i]
                save_img(results[i], outputs_root / f'{img_name}.jpg')
                np.savez(outputs_root / f'{img_name}.npz', **dump_results_i)

        get_logger(brief=True).info(f'🎨 Rendering results are under {outputs_root}.')

    get_logger(brief=True).info(f'🎊 Everything is done!')
    monitor.report()


if __name__ == '__main__':
    main()