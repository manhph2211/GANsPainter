import torch


def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    re = disc_X(real_X)
    fake = disc_X(fake_X)
    disc_loss = (adv_criterion(re,torch.ones_like(re))  + adv_criterion(fake,torch.zeros_like(fake)))/2
    return disc_loss


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    fake_Y = gen_XY(real_X)
    re = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(re,torch.ones_like(re))
    return adversarial_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    re  = gen_YX(real_X)
    identity_loss = identity_criterion(re,real_X)
    return identity_loss, re


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X,real_X)
    return cycle_loss, cycle_X


def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    fake_B = gen_AB(real_A)
    fake_A = gen_BA(real_B)
    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adversarial_loss_1,_ = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    adversarial_loss_2,_ = get_gen_adversarial_loss(real_B,disc_A, gen_BA, adv_criterion)

    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_1,f_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_2,f_B = get_identity_loss(real_B, gen_AB, identity_criterion)
    

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_loss_1,_ = get_cycle_consistency_loss(real_B,fake_A,gen_AB, cycle_criterion)
    cycle_loss_2,_ = get_cycle_consistency_loss(real_A,fake_B,gen_BA, cycle_criterion)

    # Total loss
    adversarial_loss = adversarial_loss_1 + adversarial_loss_2
    identity_loss = identity_loss_1 + identity_loss_2
    cycle_loss = cycle_loss_1 + cycle_loss_2
    gen_loss = (lambda_identity*identity_loss + lambda_cycle*cycle_loss + adversarial_loss)
    return gen_loss, fake_A, fake_B