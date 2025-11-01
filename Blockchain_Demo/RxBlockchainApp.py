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


# --------------------------
# Tampering Demonstration (for teaching)
# --------------------------
# Uncomment these lines to simulate an attack on the blockchain:
# if len(rx_chain.chain) > 1:
#     rx_chain.chain[1].transactions[0]["drug_name"] = "Fentanyl 100mg"  # malicious change
#     rx_chain.chain[1].hash = rx_chain.chain[1].compute_hash()  # recalc this block's hash only
#     st.warning("⚠️ A past block was manually tampered with for demonstration!")


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