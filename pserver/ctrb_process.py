def process(fin, fout):
    fin = open(fin, 'r')
    fout = open(fout, 'w')

    while True:
        line = fin.readline().rstrip()
        if not line:
            break

        parts = line.split()
        label, feats = parts[0], [int(feat) for feat in parts[1:]]
        feats.sort()

        fout.write(label)
        for f in feats:
            fout.write(' ')
            fout.write('%d:1' % f)
        fout.write('\n')

    fin.close()
    fout.close()

process('data/ctrb_train', 'data/ctrb_train.fixed')
process('data/ctrb_test', 'data/ctrb_test.fixed')
