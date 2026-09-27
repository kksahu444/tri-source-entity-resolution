# Error analysis: where macro F0.5 is lost (dev OOF, rule `two_threshold`)


## Loss decomposition (share of 1 - macro F0.5)

| Component | Loss | Share |
|---|---|---|
| false links | 0.00526 | 27.1% |
| model misses | 0.00971 | 50.0% |
| blocking | 0.00443 | 22.8% |

## By slice (mean loss per entity x share of entities = contribution)

| Slice | Entities | FP loss | Model-miss loss | Blocking loss |
|---|---|---|---|---|
| all | 100.0% | 0.00526 | 0.00971 | 0.00443 |
| US | 60.7% | 0.00256 | 0.00571 | 0.00210 |
| India | 39.3% | 0.00270 | 0.00400 | 0.00233 |
| true singletons | 5.6% | 0.00123 | 0.00000 | 0.00000 |
| 1 true match | 5.4% | 0.00050 | 0.00187 | 0.00072 |
| 2-3 true matches | 41.1% | 0.00230 | 0.00396 | 0.00187 |
| 4+ true matches | 47.9% | 0.00124 | 0.00388 | 0.00184 |
| name-city crowd >= 3 | 3.3% | 0.00019 | 0.00047 | 0.00057 |
| has a blocked non-Latin record | 0.3% | 0.00001 | 0.00007 | 0.00054 |
| has a blocked empty-address record | 1.5% | 0.00005 | 0.00016 | 0.00129 |

### India: false links (selected, not a match)

- `Sunrise Consultants Private Limited | No. 3094, 6Th A Main Hal 2Nd Stage, Bengaluru, Bangalore, Karnataka`  vs  `ಸನ್‌ರೈಸ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | #2692, Hal Ii Stage, Bangalore, ಕರ್ನಾಟಕ`
- `Apex Media Private Limited | House No. 13, Rajpur Road Civil Lines, Delhi, Central Delhi, Delhi`  vs  `APEX MEDIA PRÍVATE SWASTIK CARE | Hn 97 House No. 13, Rajpur Road Civil Lines, Central Delhi, Delhi, दिल्ली`
- `Asg Holdings | Amritsar, Punjab, 23-B, Amritsar, Albert Road`  vs  `Asg Oldifs | `
- `Natural & Associates | Xxii/263, A- 13, 1St Floor, C G N Building Opp.Govt Hospital, Trippunithura, Cochin, Ernakulam, Kerala`  vs  `Natural & Ass0ciates | `
- `Spellbound Foundation | No 1 Little Heart Complex, Swami Vivekananda 1St Crs, Hunasamaranahalli, Bg North, Bangalore, Karnataka`  vs  `unitedcare.com | NO 1 LITTLE HEART COMPLEX, SWAMI VIVEKANANDA 1ST CRS, HUNASAMARANAHALLI, BG NORTH, Karnataka`
- `ORA Steel Private Limited | S1, Ii Floor, Dhanya Castle, Plot 18, Jayapriya Nagar, Kolapakkam, Sriperumbudur, Kanchipuram, Tamil Nadu`  vs  `ORA Steel Private Ltd | `
- `Pranya Trucks-Anekal | # 72, Shop No 12, 1St Floor, Sms Complex, Hosur Main Road, Neraluru Village, Attibele Hobli, Anekal, Anekal, Bangalore, Karnataka`  vs  `Pranyi [Trucks-Anekal] | # A-72 , SHOP NO 12, 1ST FLOOR, SMS COMPLEX, HOSUR MAIN ROAD, NERALURU VILLAGE, ATTIBELE HOBLI, ANEKAL, ANEKAL, ಕರ್ನಾಟಕ`
- `Big Constructions Private Limited | No. 10, 1St Floor, 6Th Main A.K. Colony, Sultan Palya, R.T. Nagar Po, St, Bangalore, Karnataka`  vs  `ಬಿಗ್ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | A5-1, Bangalore, KA`
- `Abn Clinic | 2Nd Floor, East Wing, Campus 1A, Rmz Millennia Business Park, No 143, Mgr Salai, Kandanchavadi, Saidapet, Kanchipuram, Tamil Nadu`  vs  `Abn Clinic | `
- `Bangalore South India Pvt Ltd | Sy No 45, Gorvigere Village, Bangalore South, Bangalore, Karnataka`  vs  `Bangalore South Private Limited Center | ##45, Bangalore, Bangalore South, ಕರ್ನಾಟಕ`
- `Arihant Finance Private Limited | 2 Jawahar Marg Siyaganj Pp Rathor Chamber 2Ed Floor Shop Number, 205, Indore, Madhya Pradesh`  vs  `Arihant Finance Limited | NO 09 JAWAHAR MARG SIYAGANJ PP RATHOR CHAMBER 2ED FLOOR SHOP NUMBER, 205, मध्य प्रदेश`
- `Adittech Lal Limited | H No. 1974/1186, Nangla, Enclave Near Lohia Vatika, Faridabad, Haryana`  vs  `Adittekh Lal Limited | Haryana, #1974/1186- , NANGLA, ENCLAVE NEAR LOHIA VATIKA, FARIDABAD`

### India: true matches retrieved but not selected

- `Construction Tegh Consultancy | 11-2-89, 11-2-111 Komati Galli Ward No-25, Rangampet Migaddar Shorapur, Shorapur, Yadgir, Gulbarga, Karnataka`  vs  `Construction Tegh | `
- `Sky Industries Limited | H. No. 575, V.P.O. Samaspur Majra, Jhajjar, Haryana`  vs  `sky industries limited | `
- `Khurana Growth Corp | 10A, Manavillas, Ice Factory Road College Square, Cuttack, Orissa`  vs  `Shri Khurana-Growth | `
- `Suryam Cosmetics Pvt. Ltd. | M-201-D, Saurabh Vihar Hari Nagar, Badarpur, New Delhi, South Delhi, Delhi`  vs  `Fayeevo | M-201-D, Saurabh Vihar Hari Nagar, Badarpur, New Delhi, South Delhi, Delhi`
- `Universal Technology | Door No.1, 4Th Floor, No.90/2, Vrr Royal Building, 17Th Cross, 21St A Main, 1St Sector, Hsr, Layout, Bangalore, Karnataka`  vs  `Universal (Techniology) | `
- `XUS Mercantiles Private Limited | Sb-Shop No.14, Shyamwala Complex, Opp.Bhadra Kacheri Khatri Pole, Nr. Moti Chhipwad, Pani Gat, E Road, Vadodara, Gujarat`  vs  `XUS Mércantiles Private Limited | `
- `Bangalore North Productions Private Limited | 235/16, 4Th Main Road, Vyalikaval (Vkl), Bangalore North, Bangalore, Karnataka`  vs  `Private Bangalore Limited Productions North | 235/1, Bengaluru, KA`
- `Al Estate Private Limited | No.308/287, Konappana Agrahara, Hosur Main Road, Begur Hobli, Bangalore, Karnataka`  vs  `Al Estate Private Ltd | `
- `Star Vyankatesh Private Limited | No. 57, Elavarasi Mahal, Seegehalli, Whitecity Layout 6Th Main, K R Puram, Bangalore, Karnataka`  vs  `Private Star Vyacnkatesh Limited | 57, ಕರ್ನಾಟಕ, Bangalore`
- `Southern Real (India) Private Limited | Delhi, Rohini, Delhi, E3/24, Ground Floor, Sector-7`  vs  `Southern Real (India) Private | E3/2, Ground Floor, Sector-7, New Delhi, Delhi, दिल्ली`
- `Arjuna & Associates Associates | D. No. 83-151, Etukur Road, Guntur, Andhra Pradesh`  vs  `Arjuna & Associates Associates Corporation | #83-15, Etukur Road, Guntur, AP`
- `Lotus Infotech Pvt Ltd | T C 5/298 Kylasamkattela Sreekariyam P O, Thiruvananthapuram, Trivandrum, Kerala`  vs  `Fayegild | T C 5/298 Kylasamkattela Sreekariyam P O, Thiruvananthapuram, Trivandrum, Kerala`

### India: true matches never retrieved

- `Raman Traders Limited | No 16/A, Opp. Ace Designers Chokkasandra, T. Dasarahalli Post, Bengaluru, Bangalore, Karnataka`  vs  `RAMAN TRAERIS LIMITED | 16/A , OPP. ACE DESIGNERS CHOKKASANDRA, T. DASARAHALLI POS,T BENGALURU, Karnataka`
- `Bangalore North Infra Limited | No 269, Sri Venkatesvara Svami Nilayam, 7Th Cross Road, San Siri Residency Layout, Sathanur Village, Opp. Bsf Cedco Sts Campus Road, Bangalore North, Bangalore, Karnataka`  vs  `Bangalore North Infra | #269/9, Bangalore North, Bangalore, KA`
- `Star Power Private Limited | Unit No 1267, 1268, 1269 14Th Floor, B Wing, Mittal Tower, M G Ro, Ad, Bangalore, Karnataka`  vs  `ಸ್ಟಾರ್ ಪವರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | UNIT NO 1267/9, BANGALORE, Karnataka`
- `Suraj Horticulture | 45, Old Punjab Bus Stand Railway Station Fatehpuri, Delhi, Central Delhi, Delhi`  vs  `Suraj-H0rticulture | B3/45, Delhi, Central Delhi, दिल्ली`
- `Universal Tirupati Projects Private Limited | House No.339, Ward No.13 Kath Mandi Bighar Road, Fatehabad, Haryana`  vs  `Zetanyla | HN 164 HOUSE NO.339, FATEHABAD, Haryana`
- `Delhi India Group | B-26 Greater Kailash - 1, Delhi, South Delhi, Delhi`  vs  `delhiindiagroup.com | B-26 Greatre Kailash - 1, Delhi, South Delhi, दिल्ली`
- `Gold Power Private Limited | Noida, Ithum Tower, Plot No. A 40 Sector 62, Uttar Pradesh, Gautam Buddha Nagar, B-1045`  vs  `... Gold Private Limited Center | B-1047, Ithum Tower, Plot No. A 40 Sector 62, Noida, Gautam Buddha Nagar, उत्तर प्रदेश`
- `Lotus Products Pvt Ltd | 00 Fatahapurwa, Takrohi Chinhat, Lucknow, Uttar Pradesh`  vs  `Lotus Pvt Ltd Center | `
- `Ace Media Private Limited | 515-516, Topkhana Bazar, Ambala Cantt, Ambala, Haryana`  vs  `Ace Mela Private Limited | 15-516, Ambala, Ambala Cantt., HR`
- `HVH Services Private Limited | Karnataka, Bangalore, No. B-56, Bangalore, 5Th Cross, Iti Industrial Estate, Whitefield Road, Dyavasandra, Mahadevapu, Ra Post`  vs  `servicesprivate.com | No. B-d/56, Bangalore, ಕರ್ನಾಟಕ`
- `Sree Enterprises Limited | Unit No-60, No-260, Third Floor Domlur Layout, 2Nd Stage, Bangalore, Karnataka`  vs  `M/s Sree Enterprises | Unit No-60, Bangalore, KA`
- `Empower Sports Limited | Flat D-1 And D-4, 444 Min Village Asola Fatehpur Beri, New Delhi, South Delhi, Delhi`  vs  `Empower Shoarts Limited | Flat D-1 And D-4, New Delhi, DL`

### US: false links (selected, not a match)

- `Summit Cbre Center | 36171 2nd Street, Pablo, MT`  vs  `Summit Center Cbre | `
- `Hoang Frontier Occupational Therapy LLC | NY, Hempstead, 370 Hill Avenue`  vs  `Young Frontier Occupational Therapy | New York, Elmont, 370D Hill Ave`
- `Baird, Gerrie P., DDS | 80 Chautauqua Avenue, Jamestown, NY`  vs  `Baird, Gerrie P., DDS Corporation | `
- `Safe Infrastructure Holdings | 380 Pine Street, Dover-foxcroft, ME`  vs  `SAFE INFRASTRUCTURE HOLDINGS | 382 PINE ST, DOVER-FOXCROFT, ME`
- `Anderson Allied Imperii Inc. | 18011 Peralta Cove Circle, Cypress, TX`  vs  `ANDERSON ALLIED IMPERII INC. | `
- `O 4 Total Charles | 4026 Forest Teal Court, Fresno, TX`  vs  `SWW.COM | 4026 FOREST TEAL COURT, NULL, FRESNO TOWNSHIP, TX`
- `Smart Energetics, LLC | 3251 1450 Street, Herrick, IL`  vs  `Smart énergetics, LLC | `
- `Tracie Romero Marshall | 3452 Ararat Road, Ararat, NC`  vs  `Tracie Romero Marsha1l | `
- `Select Prairie Vista LLC | 18329 Bellingrath Lakes Avenue, Central City, LA`  vs  `Select Prairie LLC Partners | `
- `Doran Restaurants Inc. | 60 Broadway Street, Baileyville, ME`  vs  `Doran Studio Partners | 60 Broadway Street, Maine, Baileyvile`
- `Sanguinetti Marketing Inc. | 130 Cherry Street, Unit Apartment 408, Ithaca, NY`  vs  `Sanguinetti and Lhnv Inc | `
- `Premier Lifesciences LLC | 765 Highland Drive, Beaumont, TX`  vs  `Premier-Lifesciences | `

### US: true matches retrieved but not selected

- `Urban Massage | 2062 Keota Drive, Pueblo West, CO`  vs  `Urban Massage LLC | `
- `Patriot Committee Inc | 23206 52nd Avenue, Mountlake Terrace, WA`  vs  `Patriot Committee Trading | `
- `53/53 Copley | 30 Bishops Lane, Catonsville, MD`  vs  `Wexveo | 30 BISHOPS LN, CATONSVILLE, MD`
- `I & W Generating LLC | 205 Copper Drive, Smyth County, VA`  vs  `SYNCALOMIRA | 205 COPPER DR, RURAL RETREAT, VA`
- `Foot & Ankle Care LLC | 2 Conrad Street, Thomasville, NC`  vs  `Foot and Ankle Care LLC | `
- `Helios Technologies | 706 Maple Street, Hoxie, AR`  vs  `Helios Technologies | `
- `Marshall & Collins Iheartmedia LP | 7283 Barton Hill Drive, Parma, OH`  vs  `Marshall & Collins | `
- `Interfaith Great Charities | 413 Southland Street, Hendersonville, TN`  vs  `Interfaith Great | `
- `Kathi Mendoza Broadband LLC | VA, Hanover County, 10984 Founders Place`  vs  `bkmendoza.com (ID: 69290) | Virginia, 10984 Founders Place, Hanover County`
- `Jones Midwest Transition Inc. | 10235 Birchbrook Drive, St. John, IN`  vs  `JONES MIDWEST TRANSITION INCORPORATED | `
- `American Society | 2612 1B Guyer Street, High Point, NC`  vs  `American Society Inc. | `
- `Elita Dean Coastal Allegro L.L.C. | 215 C Street, Unit 207, Washington, DC`  vs  `elita dean coastal allegro l.l.c. | `

### US: true matches never retrieved

- `Desert Group | 359 Burley Road, Collierville, TN`  vs  `Desert Associates | `
- `Pediatric Dentistry Clinic | 261 Jefferson Street, North Wilkesboro, NC`  vs  `Pediatric Dentistry | Jefferson Street, North Wilkesbooro, North Carolina`
- `Summit Inc | 508 Davidson Street, Unit 216, Nashville, TN`  vs  `Summit Incorporated Trading | `
- `Pacific Holding Company, L.L.C. | TN, Springfield, 6947 Cowan Road`  vs  `Pacific Company, L.L.C. Service | `
- `Desert Arrow, LLC | 193 Teva Hollow Lane, Taylorsville, NC`  vs  `DESERT ADNOB,W [LLC] | `
- `White Evolution L.L.C. | 11207 First Street, Unit 2S, Mokena, IL`  vs  `White Eevlnofulcron L.L.C. | `
- `Internal Medicine Physicians LLC | 38 6, Somers, NY`  vs  `#internalmedicine | 38 6, GRANITE SPRRINGS, NY`
- `Capital Earth LLC | NY, 24 81, Barker`  vs  `CAPITA1 EARTH LLC | 24 81, PO BOX 2277, CASTLE CREEK, NY`
- `Reon LLC | 108 2nd Street, Chalmers, IN`  vs  `Ro LLC | 108 SECOND SAINT, PMB 3265, CHALMERS, IN`
- `Halcyon Corp | 838 27th Street, Dickinson, ND`  vs  `Hlctdhn Corp | 838 27RD STREET, DCIKINSON, ND`
- `Meridian Inc | 318 Saddle Brook Drive, Lebanon, TN`  vs  `Meridian Inc | `
- `Classic Patriot Trust Group | 1363 15th Street, Dickinson, ND`  vs  `Classic Patriot Group Services | 15TH STREET, DICKNISON, ND`
