import torch


def train_embeddings(model, trainloader, validloader, device, args, tb):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    clip = lambda x: x.where(x <= 4, torch.tensor([4], dtype=torch.long))
    best_loss = 9.9

    model.bat_emb.weight.requires_grad = False
    model.pit_emb.weight.requires_grad = False
    model.team_emb.weight.requires_grad = False

    for epoch in range(50):
        print('Epoch', epoch)
        # Training
        model.train()
        sum_losses = 0.0
        for start_obs, _, info, targets in trainloader:
            fld_team_id = torch.where(start_obs['bat_home_id'] < 0.5,
                                      info['home_team_id'],
                                      info['away_team_id'])
            bat_dest, run1_dest, run2_dest, run3_dest = model.embed(
                start_obs['outs_ct'].to(device),
                start_obs['pit_id'].to(device),
                fld_team_id.to(device),
                start_obs['bat_id'].to(device),
                start_obs['base1_run_id'].to(device),
                start_obs['base2_run_id'].to(device),
                start_obs['base3_run_id'].to(device)
            )
            loss = \
                loss_fn(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                loss_fn(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                loss_fn(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                loss_fn(run3_dest, clip(targets['run3_dest']).squeeze().to(device))
            sum_losses += bat_dest.shape[0] * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = sum_losses / len(trainloader.dataset)
        tb.add_scalar('train loss', mean_loss, epoch)

        # Validation
        model.eval()
        sum_losses = 0.0
        for start_obs, _, info, targets in validloader:
            fld_team_id = torch.where(start_obs['bat_home_id'] < 0.5,
                                      info['home_team_id'],
                                      info['away_team_id'])
            bat_dest, run1_dest, run2_dest, run3_dest = model.embed(
                start_obs['outs_ct'].type(torch.long).to(device),
                start_obs['pit_id'].to(device),
                fld_team_id.to(device),
                start_obs['bat_id'].to(device),
                start_obs['base1_run_id'].to(device),
                start_obs['base2_run_id'].to(device),
                start_obs['base3_run_id'].to(device)
            )
            loss = \
                loss_fn(bat_dest, clip(targets['bat_dest']).squeeze().to(device)) + \
                loss_fn(run1_dest, clip(targets['run1_dest']).squeeze().to(device)) + \
                loss_fn(run2_dest, clip(targets['run2_dest']).squeeze().to(device)) + \
                loss_fn(run3_dest, clip(targets['run3_dest']).squeeze().to(device))
            sum_losses += bat_dest.shape[0] * loss.item()
        mean_loss = sum_losses / len(validloader.dataset)
        tb.add_scalar('valid loss', mean_loss, epoch)

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), 'models/model.pt')
            print('Model saved.')

        # Weight histograms
        tb.add_histogram('bat emb', model.bat_emb.weight, epoch)
        tb.add_histogram('pit emb', model.pit_emb.weight, epoch)
        tb.add_histogram('team emb', model.team_emb.weight, epoch)


def load_embeddings(model, tb):
    state_dict = torch.load('models/embeddings.pt')
    with torch.no_grad():
        model.bat_emb.weight.copy_(state_dict['bat_emb.weight'])
        model.pit_emb.weight.copy_(state_dict['pit_emb.weight'])
        model.team_emb.weight.copy_(state_dict['team_emb.weight'])

    tb.add_histogram('bat emb', model.bat_emb.weight, 0)
    tb.add_histogram('pit emb', model.pit_emb.weight, 0)
    tb.add_histogram('team emb', model.team_emb.weight, 0)
