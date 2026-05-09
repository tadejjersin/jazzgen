import pretty_midi
import numpy as np

def get_piano_roll(midi_path, fs=100):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # get piano roll
    piano_roll = pm.instruments[0].get_piano_roll(fs=fs)

    # crop to 88 piano keys (MIDI 21–108, A0 to C8)
    piano_roll = piano_roll[21:109, :]

    # normalize to [0, 1] (127 is the max volume)
    piano_roll = np.clip(piano_roll / 127.0, 0.0, 1.0)

    return piano_roll


class Tokenizer:
    def __init__(self, time_step=0.01, max_shift=100, velocity_bins=32):
        self.time_step = time_step
        self.max_shift = max_shift
        self.velocity_bins = velocity_bins

        self.min_pitch = 21 # A0
        self.max_pitch = 108 # C8
        self.num_pitches = 88

        # vocabulary layout
        self.note_on_offset = 0                          # 0–87
        self.note_off_offset = 88                        # 88–175
        self.time_shift_offset = 176                     # 176–275
        self.velocity_offset = 176 + max_shift           # 276–307
        self.end_token = self.velocity_offset + velocity_bins  # last token

        self.vocab_size = self.end_token + 1


    # helpers
    def quantize_time(self, t):
        return int(round(t / self.time_step))

    def velocity_to_bin(self, velocity):
        bin = int(velocity / 128 * self.velocity_bins)
        return min(bin, self.velocity_bins - 1)

    def bin_to_velocity(self, bin):
        return int((bin + 0.5) / self.velocity_bins * 127)
    
    def token_to_string(self, token):
        if token == self.end_token:
            return "END"

        elif token >= self.velocity_offset:
            vel_bin = token - self.velocity_offset
            return f"VELOCITY_{vel_bin}"

        elif token >= self.time_shift_offset:
            shift = token - self.time_shift_offset + 1
            return f"TIME_SHIFT_{shift}"

        elif token >= self.note_off_offset:
            pitch = token - self.note_off_offset + self.min_pitch
            return f"NOTE_OFF_{pitch}"

        else:
            pitch = token + self.min_pitch
            return f"NOTE_ON_{pitch}"
        
    def clean_overlapping_same_pitch_notes(self, notes, min_duration_steps=1):
        notes_by_pitch = {}
        for note in notes:
            if self.min_pitch <= note.pitch <= self.max_pitch:
                notes_by_pitch.setdefault(note.pitch, []).append(note)

        cleaned_notes = []

        for pitch, pitch_notes in notes_by_pitch.items():
            # sort notes of the same pitch by start time, then end time
            pitch_notes = sorted(pitch_notes, key=lambda n: (n.start, n.end))

            for i in range(len(pitch_notes) - 1):
                current_note = pitch_notes[i]
                next_note = pitch_notes[i + 1]

                # if the current note overlaps the next note of the same pitch,
                # shorten the current note.
                if next_note.start < current_note.end:
                    current_note.end = next_note.start

            # keep only notes that still have positive duration after cleaning
            for note in pitch_notes:
                start_step = self.quantize_time(note.start)
                end_step = self.quantize_time(note.end)

                if end_step - start_step >= min_duration_steps:
                    cleaned_notes.append(note)

        # sort all notes globally again
        cleaned_notes.sort(key=lambda n: (n.start, n.pitch, n.end))

        return cleaned_notes

    # tokenization
    def tokenize(self, midi_path):
        pm = pretty_midi.PrettyMIDI(midi_path)
        events = []

        notes = pm.instruments[0].notes # piano only 

        # clean notes
        notes = self.clean_overlapping_same_pitch_notes(notes)

        # collect note events
        for note in notes:
            if self.min_pitch <= note.pitch <= self.max_pitch:
                pitch = note.pitch - self.min_pitch
                start = self.quantize_time(note.start)
                end = self.quantize_time(note.end)
                vel_bin = self.velocity_to_bin(note.velocity)

                # skip notes that disappear after quantization
                if end <= start:
                    continue

                events.append((start, "on", pitch, vel_bin))
                events.append((end, "off", pitch, None))

        # sort by time (NOTE_OFF before NOTE_ON if equal time)
        events.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))

        tokens = []
        current_time = 0

        for time, typ, pitch, vel_bin in events:
            delta = time - current_time

            # Encode TIME_SHIFT (can be multiple TIME_SHIFT-s in a row if long pause)
            while delta > 0:
                shift = min(delta, self.max_shift)
                tokens.append(self.time_shift_offset + shift - 1)
                delta -= shift

            current_time = time

            if typ == "on":
                # VELOCITY must come before NOTE_ON
                tokens.append(self.velocity_offset + vel_bin)
                tokens.append(self.note_on_offset + pitch)
            else:
                tokens.append(self.note_off_offset + pitch)

        # add END token
        tokens.append(self.end_token)

        return tokens
        


    # detokenization
    def detokenize(self, tokens, output_path=None):
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        current_time = 0.0
        active_notes = {}
        current_velocity = 80  # default

        for token in tokens:
            if token == self.end_token:
                break

            # VELOCITY
            elif token >= self.velocity_offset:
                vel_bin = token - self.velocity_offset
                current_velocity = self.bin_to_velocity(vel_bin)

            # TIME_SHIFT
            elif token >= self.time_shift_offset:
                shift = token - self.time_shift_offset + 1
                current_time += shift * self.time_step

            # NOTE_OFF
            elif token >= self.note_off_offset:
                pitch = token - self.note_off_offset + self.min_pitch
                if pitch in active_notes:
                    start, velocity = active_notes[pitch]
                    if current_time > start:
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start,
                            end=current_time
                        )
                        instrument.notes.append(note)
                    del active_notes[pitch]

            # NOTE_ON
            else:
                pitch = token + self.min_pitch
                active_notes[pitch] = (current_time, current_velocity)

        pm.instruments.append(instrument)
        if output_path is not None:
            pm.write(output_path)
        return pm