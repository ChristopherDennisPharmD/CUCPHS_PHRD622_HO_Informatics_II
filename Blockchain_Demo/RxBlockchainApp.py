# Pharmacy Blockchain Streamlit Application

# Import libraries/modules
import streamlit as st
import hashlib
import json
from time import time

# Define classes for blocks and blockchains

class Block:
    def __init__(self, index, transactions, timestamp, previous_hash):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()
    
    def compute_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.unconfirmed_txns = []
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, [], time(), "0")
        self.chain.append(genesis_block)
    
    def last_block(self):
        return self.chain[-1]
    
    def add_transaction(self, prescription):
        self.unconfirmed_txns.append(prescription)
    
    def mine_block(self):
        if not self.unconfirmed_txns:
            return None
        last_block = self.last_block()
        new_block = Block(
            index = last_block.index + 1,
            transactions = self.unconfirmed_txns,
            timestamp = time(),
            previous_hash = last_block.hash
        )
        self.chain.append(new_block)
        self.unconfirmed_txns = []
        return new_block
    
# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="ePrescription Blockchain Demo", layout="wide")

st.title("💊 ePrescription Blockchain Demo")
st.markdown("""
This interactive demo shows how **blockchain technology** can be used to 
securely record a history of prescriptions written across a community.
Each mined block contains prescription transactions that are **linked by hashes**, 
creating an immutable ledger.
""")

# Persistent blockchain instance (stored in Streamlit session)
if "rx_chain" not in st.session_state:
    st.session_state.rx_chain = Blockchain()

rx_chain = st.session_state.rx_chain


# -------------------------
# Input Form
# -------------------------
st.subheader("🧾 Write a New Prescription")

with st.form("new_rx_form"):
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", "P12345")
        prescriber_id = st.text_input("Prescriber ID", "MD6789")
        drug_name = st.text_input("Drug Name", "Atorvastatin 20 mg")
    with col2:
        quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=30)
        date_written = st.date_input("Date Written")
        pharmacy_id = st.text_input("Pharmacy ID", "RX001")
    notes = st.text_area("Notes", "Take one tablet every night at bedtime")

    submitted = st.form_submit_button("Add to Pending Transactions")

if submitted:
    rx_chain.add_transaction({
        "patient_id": patient_id,
        "prescriber_id": prescriber_id,
        "drug_name": drug_name,
        "quantity": quantity,
        "date_written": str(date_written),
        "pharmacy_id": pharmacy_id,
        "notes": notes
    })
    st.success("✅ Prescription added to pending transactions.")

# -------------------------
# Pending Transactions
# -------------------------
if rx_chain.unconfirmed_txns:
    st.subheader("⏳ Pending Transactions")
    st.json(rx_chain.unconfirmed_txns)
    if st.button("⛏️ Mine Block"):
        new_block = rx_chain.mine_block()
        st.success(f"✅ Mined new block #{new_block.index} with hash: {new_block.hash}")
else:
    st.info("No pending prescriptions. Add one above to mine a new block.")

# --------------------------
# Display the Blockchain
# --------------------------
st.subheader("🔗 Blockchain Ledger")

for block in rx_chain.chain:
    with st.expander(f"Block #{block.index} - Hash: {block.hash[:15]}..."):
        st.write(f"**Timestamp:** {block.timestamp}")
        st.write(f"**Previous Hash:** {block.previous_hash}")
        st.json(block.transactions)


### TODO: Need to fix the tampering/repair demonstration code.
# --------------------------
# Tampering Demonstration (Interactive + Repair)
# --------------------------

# st.subheader("🧪 Tamper Mode Demonstration")

# st.markdown("""
# Use this section to simulate tampering and repair of the blockchain.
# - **Tamper Mode** changes a past prescription (Block #1).
# - **Repair Chain** re-mines each block sequentially to restore integrity.
# """)

# # Initialize a tampering flag in session state
# if "tampered" not in st.session_state:
#     st.session_state.tampered = False

# col_tamper, col_repair = st.columns(2)

# with col_tamper:
#     tamper_mode = st.checkbox("🔧 Enable Tamper Mode (for demo)", value=st.session_state.tampered)

#     if tamper_mode and not st.session_state.tampered:
#         # Only tamper once per activation
#         if len(rx_chain.chain) > 1:
#             original_drug = rx_chain.chain[1].transactions[0]["drug_name"]
#             rx_chain.chain[1].transactions[0]["drug_name"] = "TamperedDrug 999mg"
#             rx_chain.chain[1].hash = rx_chain.chain[1].compute_hash()
#             st.session_state.tampered = True
#             st.warning(
#                 f"⚠️ Block #1 has been **tampered**! "
#                 f"Original drug: '{original_drug}' → Now: 'TamperedDrug 999mg'"
#             )
#         else:
#             st.info("Not enough blocks yet. Please mine at least one new block first.")
#     elif not tamper_mode and st.session_state.tampered:
#         # Turning tamper mode off just toggles the flag (doesn't auto-repair)
#         st.session_state.tampered = True  # keep flag until repaired
#         st.info("Tamper mode turned off. Use 'Repair Chain' below to fix integrity.")
#     else:
#         st.info("Tamper mode is off — blockchain currently unmodified.")

# # --------------------------
# # Chain Repair Function
# # --------------------------
# def repair_chain(blockchain):
#     """Recompute hashes from genesis to restore continuity."""
#     for i in range(1, len(blockchain.chain)):
#         blockchain.chain[i].previous_hash = blockchain.chain[i - 1].hash
#         blockchain.chain[i].hash = blockchain.chain[i].compute_hash()

# with col_repair:
#     if st.button("🛠️ Repair Chain"):
#         repair_chain(rx_chain)
#         st.session_state.tampered = False
#         st.success("✅ Chain repaired! All hashes recomputed to restore integrity.")



# --------------------------
# Verification
# --------------------------
st.subheader("🧩 Verify Chain Integrity")

def verify_chain(chain):
    for i in range(1, len(chain)):
        prev = chain[i-1]
        curr = chain[i]
        if curr.previous_hash != prev.hash:
            return False
    return True

if verify_chain(rx_chain.chain):
    st.success("✅ Blockchain is valid and untampered.")
else:
    st.error("⚠️ Blockchain has been tampered with!")