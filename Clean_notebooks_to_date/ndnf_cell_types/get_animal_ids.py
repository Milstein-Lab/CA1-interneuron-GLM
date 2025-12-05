import h5py
import numpy as np
import os

# ---------------------------------------------------------------------
# 1. HDF5 helpers
# ---------------------------------------------------------------------

def decode_hdf5_string_dataset(ds):
    """
    Decode a MATLAB-style uint16 char dataset into a Python string.
    Assumes 'ds' is a 1D or 2D uint16 array with zero padding.
    """
    arr = ds[...].flatten()
    arr = arr[arr != 0]      # drop any zero padding
    return ''.join(chr(int(c)) for c in arr)


def load_session_table(mat_filepath, sessions_key="sessions"):
    """
    Load the 'sessions' table from the MATLAB HDF5 file and return:
      - session_ids: 1D np.array of strings (row 0)
      - track_types: 1D np.array of strings (row 1)
    """
    with h5py.File(mat_filepath, 'r') as f:
        sessions = f[sessions_key]
        n_rows, n_cols = sessions.shape

        session_strings = np.empty((n_rows, n_cols), dtype=object)

        for i in range(n_rows):
            for j in range(n_cols):
                ref = sessions[i, j]
                ds = f[ref]
                session_strings[i, j] = decode_hdf5_string_dataset(ds)

    # row 0: session IDs (e.g. "CG189_001")
    # row 1: track types (e.g. "E0", "A1", "B1")
    session_ids = session_strings[0, :]
    track_types = session_strings[1, :]

    return np.array(session_ids), np.array(track_types)


# ---------------------------------------------------------------------
# 2. Group by animal and track type
# ---------------------------------------------------------------------

def build_animal_session_dict(session_ids, track_types):
    """
    Build a dict:
        { animal_id: [ {'idx': session_index, 'session_id': str, 'track': str}, ... ] }
    where 'idx' is the 0-based session index.
    """
    animal_ids = np.array([sid.split('_')[0] for sid in session_ids])

    animal_sessions = {}
    for idx, (animal, sid, track) in enumerate(zip(animal_ids, session_ids, track_types)):
        animal_sessions.setdefault(animal, []).append({
            "idx": idx,
            "session_id": sid,
            "track": track,
        })

    # Sort each animal's sessions by global index (chronological order)
    for animal in animal_sessions:
        animal_sessions[animal].sort(key=lambda x: x["idx"])

    return animal_sessions


# ---------------------------------------------------------------------
# 3. Build dicts for: both A1 & B1, B1 only, A1 only
# ---------------------------------------------------------------------

def build_a1_b1_dicts(animal_sessions):
    """
    Returns three dicts, each mapping animal -> list of 0-based session indices:
        both_a1_b1_dict  : animals that have at least one A1 and at least one B1
        b1_no_a1_dict    : animals that have B1 but never A1
        a1_no_b1_dict    : animals that have A1 but never B1
    All sessions for each animal are included (same style as your example).
    """
    both_a1_b1_dict = {}
    b1_no_a1_dict = {}
    a1_no_b1_dict = {}

    for animal, sess_list in animal_sessions.items():
        tracks = [s["track"] for s in sess_list]
        has_a1 = any(t == "A1" for t in tracks)
        has_b1 = any(t == "B1" for t in tracks)

        session_indices = [s["idx"] for s in sess_list]  # all sessions for that animal

        if has_a1 and has_b1:
            both_a1_b1_dict[animal] = session_indices
        elif has_b1 and not has_a1:
            b1_no_a1_dict[animal] = session_indices
        elif has_a1 and not has_b1:
            a1_no_b1_dict[animal] = session_indices
        # animals with neither A1 nor B1 are ignored

    return both_a1_b1_dict, b1_no_a1_dict, a1_no_b1_dict


# ---------------------------------------------------------------------
# 4. For animals with both A1 & B1, figure out who came first
# ---------------------------------------------------------------------

def build_order_lists(both_a1_b1_dict, track_types):
    """
    For each animal that saw both A1 & B1:
      - find the earliest A1 index and earliest B1 index for that animal
      - classify animal as 'B1 first' or 'A1 first'
    Then build these four lists of session indices (0-based):

        b1_first_b1_indices: all B1 session indices from animals where B1 came first
        a1_first_a1_indices: all A1 session indices from animals where A1 came first
        b1_when_a1_first   : B1 indices from animals where A1 came first
        a1_when_b1_first   : A1 indices from animals where B1 came first

    The lists are sorted and unique (same style as your example arrays).
    """
    b1_first_b1_indices = []
    a1_first_a1_indices = []
    b1_when_a1_first = []
    a1_when_b1_first = []

    for animal, session_indices in both_a1_b1_dict.items():
        # which of this animal's sessions are A1 / B1?
        a1_idxs = [idx for idx in session_indices if track_types[idx] == "A1"]
        b1_idxs = [idx for idx in session_indices if track_types[idx] == "B1"]

        if not a1_idxs or not b1_idxs:
            # should not happen given both_a1_b1_dict definition, but be safe
            continue

        first_a1 = min(a1_idxs)
        first_b1 = min(b1_idxs)

        if first_b1 < first_a1:
            # B1-first animal
            b1_first_b1_indices.extend(b1_idxs)
            a1_when_b1_first.extend(a1_idxs)
        else:
            # A1-first animal
            a1_first_a1_indices.extend(a1_idxs)
            b1_when_a1_first.extend(b1_idxs)

    # make them unique & sorted like your example
    b1_first_b1_indices = sorted(set(b1_first_b1_indices))
    a1_first_a1_indices = sorted(set(a1_first_a1_indices))
    b1_when_a1_first = sorted(set(b1_when_a1_first))
    a1_when_b1_first = sorted(set(a1_when_b1_first))

    return b1_first_b1_indices, a1_first_a1_indices, b1_when_a1_first, a1_when_b1_first


# ---------------------------------------------------------------------
# 5. One convenience wrapper to run everything for the new file
# ---------------------------------------------------------------------

def analyze_ndnf_e0a1b1(
    base_path="/Users/michaelfinch/CA1-interneuron-GLM",
    mat_name="NDNF_E0A1B1_251107",
):
    """
    Full pipeline for your new dataset.
    Returns:
      - both_a1_b1_dict
      - b1_no_a1_dict
      - a1_no_b1_dict
      - b1_first_b1_indices
      - a1_first_a1_indices
      - b1_when_a1_first
      - a1_when_b1_first
    """
    mat_filepath = os.path.join(base_path, "datasets", mat_name + ".mat")

    session_ids, track_types = load_session_table(mat_filepath)
    animal_sessions = build_animal_session_dict(session_ids, track_types)

    both_a1_b1_dict, b1_no_a1_dict, a1_no_b1_dict = build_a1_b1_dicts(animal_sessions)

    (
        b1_first_b1_indices,
        a1_first_a1_indices,
        b1_when_a1_first,
        a1_when_b1_first,
    ) = build_order_lists(both_a1_b1_dict, track_types)

    return (
        both_a1_b1_dict,
        b1_no_a1_dict,
        a1_no_b1_dict,
        b1_first_b1_indices,
        a1_first_a1_indices,
        b1_when_a1_first,
        a1_when_b1_first,
    )


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    (
        both_a1_b1_dict,
        b1_no_a1_dict,
        a1_no_b1_dict,
        b1_first_b1_indices,
        a1_first_a1_indices,
        b1_when_a1_first,
        a1_when_b1_first,
    ) = analyze_ndnf_e0a1b1()

    print("#both A1 and B1")
    for k, v in both_a1_b1_dict.items():
        print(f"{k}: {v}")

    print("\n#B1 no A1")
    for k, v in b1_no_a1_dict.items():
        print(f"{k}: {v}")

    print("\n#A1 no B1")
    for k, v in a1_no_b1_dict.items():
        print(f"{k}: {v}")

    print("\nLists:")
    print("b1_first_b1_indices =", b1_first_b1_indices)
    print("a1_first_a1_indices =", a1_first_a1_indices)
    print("b1_when_a1_first   =", b1_when_a1_first)
    print("a1_when_b1_first   =", a1_when_b1_first)
