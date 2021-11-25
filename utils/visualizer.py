from IPython import display


def visualize(dummy_batch):
    index = 0

    waveform = dummy_batch.waveform[index][:dummy_batch.waveforn_length[index]]
    durations = dummy_batch.durations[index][:dummy_batch.token_lengths[index]]

    # scale by waveform domain
    durations = durations * dummy_batch.waveforn_length[index]
    durations = durations.cumsum(dim=0).int()

    print(dummy_batch.transcript[index])
    left = 0
    for right, char in zip(durations[:10], dummy_batch.transcript[index]):
        print(char)
        display.display(display.Audio(waveform[left:right], rate=22050))
        left = right
        print('-' * 99)
