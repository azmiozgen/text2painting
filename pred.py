import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model file to load')
    args = parser.parse_args()
    model_file = args.model

    ## Data loader
    print("Data loader initializing..")
    val_dataset = TextArtDataLoader(CONFIG, mode='val')
    val_align_collate = AlignCollate(CONFIG, 'val')
    # val_batch_sampler = ImageBatchSampler(CONFIG, mode='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=CONFIG.BATCH_SIZE,
                            shuffle=False,
                            num_workers=CONFIG.N_WORKERS,
                            pin_memory=True,
                            collate_fn=val_align_collate,
                            # sampler=val_batch_sampler,
                            drop_last=True,
                            )
    print("\tValidation size:", len(val_dataset))
    n_val_batch = len(val_dataset) // CONFIG.BATCH_SIZE
    time.sleep(0.5)

    ## Init model with G
    print("\nModel initializing..")
    model = GANModel(CONFIG, model_file=model_file, mode='val', reset_lr=False)
    time.sleep(1.0)

    start = time.time()

    total_loss_g = 0.0
    total_loss_g_refiner = 0.0
    total_loss_g_refiner2 = 0.0

    data_loader = val_loader
    n_batch = n_val_batch
    model.G.eval()
    model.G_refiner.eval()
    model.G_refiner2.eval()
    train_G = False

    for i, data in enumerate(data_loader):
        iteration = i

        ## Get data
        real_first_images, real_second_images, real_images, real_wvs, fake_wvs = data
        batch_size = real_images.size()[0]

        ## Forward G
        real_wvs_flat = real_wvs.view(batch_size, -1)
        fake_images = model.forward(model.G, real_wvs_flat)

        ## Forward G_refiner
        refined1 = model.forward(model.G_refiner, fake_images)
        
        ## Forward G_refiner2
        refined2 = model.forward(model.G_refiner2, refined1)

        ## Update total loss
        loss_g, _, loss_g_refiner, _, loss_g_refiner2, _, _, _, _, _ = model.get_losses()
        total_loss_g += loss_g
        total_loss_g_refiner += loss_g_refiner
        total_loss_g_refiner2 += loss_g_refiner2

        ## Get D accuracy
        acc_rr, acc_rf, acc_fr, acc_decider_rr, acc_decider_fr, acc_decider2_rr, acc_decider2_fr = model.get_D_accuracy()
        total_acc_rr += acc_rr
        total_acc_rf += acc_rf
        total_acc_fr += acc_fr
        total_acc_decider_rr += acc_decider_rr
        total_acc_decider_fr += acc_decider_fr
        total_acc_decider2_rr += acc_decider2_rr
        total_acc_decider2_fr += acc_decider2_fr

        ## Save logs
        if iteration % CONFIG.N_LOG_BATCH == 0:
            log_tuple = phase, epoch, iteration, loss_g, loss_d, loss_g_refiner, loss_d_decider, loss_g_refiner2, loss_d_decider2,\
                            acc_rr, acc_rf, acc_fr, acc_decider_rr, acc_decider_fr, acc_decider2_rr, acc_decider2_fr
            model.save_logs(log_tuple)

        # Print logs
        if i % CONFIG.N_PRINT_BATCH == 0:
            print("\t\tBatch {: 4}/{: 4}:".format(i, n_batch), end=' ')
            if CONFIG.GAN_LOSS1 == 'wgangp':
                print("G loss: {:.4f} | D loss: {:.4f}".format(loss_g, loss_d), end=' ')
                print("| G refiner loss: {:.4f} | D decider loss {:.4f}".format(loss_g_refiner, loss_d_decider), end=' ')
                print("| G refiner2 loss: {:.4f} | D decider2 loss {:.4f}".format(loss_g_refiner2, loss_d_decider2), end=' ')
                print("| GP loss fake-real: {:.4f}".format(loss_gp_fr), end=' ')
                print("| GP loss real-fake: {:.4f}".format(loss_gp_rf), end=' ')
                print("| GP loss fake refined1-fake: {:.4f}".format(loss_gp_decider_fr), end=' ')
                print("| GP loss fake refined2-fake: {:.4f}".format(loss_gp_decider2_fr))
            else:
                print("G loss: {:.4f} | D loss: {:.4f}".format(loss_g, loss_d), end=' ')
                print("| G refiner loss: {:.4f} | D decider loss {:.4f}".format(loss_g_refiner, loss_d_decider), end=' ')
                print("| G refiner2 loss: {:.4f} | D decider2 loss {:.4f}".format(loss_g_refiner2, loss_d_decider2))
            print("\t\t\tAccuracy D real-real: {:.4f} | real-fake: {:.4f} | fake-real {:.4f}".format(acc_rr, acc_rf, acc_fr))
            print("\t\t\tAccuracy D decider real-real: {:.4f} | fake refined1-real {:.4f}".format(acc_decider_rr, acc_decider_fr))
            print("\t\t\tAccuracy D decider2 real-real: {:.4f} | fake refined2-real {:.4f}".format(acc_decider2_rr, acc_decider2_fr))

        ## Save visual outputs
        try:
            if iteration % CONFIG.N_SAVE_VISUALS_BATCH == 0 and phase == 'val':
                output_filename = "{}_{:04}_{:08}.png".format(model.model_name, epoch, iteration)
                grid_img_pil = model.generate_grid(real_wvs, fake_images, refined1, refined2, real_images, train_dataset.word2vec_model)
                model.save_img_output(grid_img_pil, output_filename)
                # model.save_grad_output(output_filename)
        except Exception as e:
            print('Grid image generation failed.', e, 'Passing.')

    total_loss_g /= (i + 1)
    total_loss_d /= (i + 1)
    total_loss_g_refiner /= (i + 1)
    total_loss_d_decider /= (i + 1)
    total_loss_g_refiner2 /= (i + 1)
    total_loss_d_decider2 /= (i + 1)
    total_loss_gp_fr /= (i + 1)
    total_loss_gp_rf /= (i + 1)
    total_loss_gp_decider_fr /= (i + 1)
    total_loss_gp_decider2_fr /= (i + 1)
    total_acc_rr /= (i + 1)
    total_acc_rf /= (i + 1)
    total_acc_fr /= (i + 1)
    total_acc_decider_rr /= (i + 1)
    total_acc_decider_fr /= (i + 1)
    total_acc_decider2_rr /= (i + 1)
    total_acc_decider2_fr /= (i + 1)
    if CONFIG.GAN_LOSS1 == 'wgangp':
        print("\t\t{p} G loss: {:.4f} | {p} D loss: {:.4f}".format(total_loss_g, total_loss_d, p=phase.title()), end=' ')
        print("| {p} G refiner loss: {:.4f} | {p} D decider loss: {:.4f}".format(total_loss_g_refiner, total_loss_d_decider, p=phase.title()), end=' ')
        print("| {p} G refiner2 loss: {:.4f} | {p} D decider2 loss: {:.4f}".format(total_loss_g_refiner2, total_loss_d_decider2, p=phase.title()), end=' ')
        print("| GP loss fake-real: {:.4f}".format(total_loss_gp_fr), end=' ')
        print("| GP loss real-fake: {:.4f}".format(total_loss_gp_rf), end=' ')
        print("| GP loss real refined1-fake: {:.4f}".format(total_loss_gp_decider_fr), end=' ')
        print("| GP loss real refined2-fake: {:.4f}".format(total_loss_gp_decider2_fr))
    else:
        print("\t\t{p} G loss: {:.4f} | {p} D loss: {:.4f}".format(total_loss_g, total_loss_d, p=phase.title()), end=' ')
        print("\t\t{p} G refiner loss: {:.4f} | {p} D decider loss: {:.4f}".format(total_loss_g_refiner, total_loss_d_decider, p=phase.title()))
        print("\t\t{p} G refiner2 loss: {:.4f} | {p} D decider2 loss: {:.4f}".format(total_loss_g_refiner2, total_loss_d_decider2, p=phase.title()))
    print("\t\tAccuracy D real-real: {:.4f} | real-fake: {:.4f} | fake-real {:.4f}".format(total_acc_rr, total_acc_rf, total_acc_fr))
    print("\t\tAccuracy D decider real-real: {:.4f} | fake refined1-real {:.4f}".format(total_acc_decider_rr, total_acc_decider_fr))
    print("\t\tAccuracy D decider2 real-real: {:.4f} | fake refined2-real {:.4f}".format(total_acc_decider2_rr, total_acc_decider2_fr))
    print("\t{} time: {:.2f} seconds".format(phase.title(), time.time() - phase_start))

## Update lr
model.update_lr(total_loss_g, total_loss_d, total_loss_g_refiner, total_loss_d_decider, total_loss_g_refiner2, total_loss_d_decider2)

## Save model
if epoch % CONFIG.N_SAVE_MODEL_EPOCHS == 0:
    model.save_model_dict(epoch, iteration, total_loss_g, total_loss_d,\
                            total_loss_g_refiner, total_loss_d_decider, total_loss_g_refiner2, total_loss_d_decider2)