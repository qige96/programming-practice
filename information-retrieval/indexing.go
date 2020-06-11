

interface Listing {
    AddDoc(docId int)
}

interface Index {
    GetListing(term string) Listing
}


