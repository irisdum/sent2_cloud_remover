"""
Functions for admin access to the SARA API.
"""

from __future__ import print_function, division

import sys
import re
import requests
isPython3 = (sys.version_info.major == 3)
if isPython3:
    from urllib.parse import urljoin
else:
    from urlparse import urljoin


def postToSara(finalXmlFile,saraurl,username, password, verbose=False, update=False):
    """
    post a xml metadata file to SARA.
    If update is True, will delete and re-post a product if it's already in the database.
    """
    with open(finalXmlFile) as mf:
        xmlStr = mf.read()
        
    r = requests.post(saraurl, data=xmlStr, auth=(username, password))
    try:
        response = r.json()
    except ValueError:
        raise saraAdminError("Posting {} to {} failed with response text: {}".format(finalXmlFile, saraurl, r.text()))

    # successful
    if "status" in response:
        if response["status"]=="success":
            if verbose: print("{}: {}".format(saraurl, response["message"]))
    elif "ErrorMessage" in response:
        if "already in database" in response["ErrorMessage"] and update:
            # delete and re-post
            try:
                identifer = re.match("Feature (\S+) already in database",response["ErrorMessage"]).group(1)
            except Exception as e:
                raise saraAdminError("Posting {} to {} failed, can't understand response: {}".format(finalXmlFile, saraurl, response["ErrorMessage"]))
            # now try delete
            try:
                deleteFromSara(identifer, saraurl, username, password, verbose=verbose)
            except saraAdminError as e:
                raise saraAdminError(e)
            postToSara(finalXmlFile, saraurl, username, password, verbose=verbose, update=False)
        else:
            raise saraAdminError("Posting {} to {} failed: {}".format(finalXmlFile, saraurl, response["ErrorMessage"]))
    else:
        # shouldn't end up here?:
        raise saraAdminError("Posting {} to {} failed with response: {}".format(finalXmlFile, saraurl, response))

           
def deleteFromSara(identifier, saraurl, username, password, verbose=False):
    """
    delete a product from SARA.
    identifier is the UUID asigned by SARA.
    """
    producturl = urljoin(saraurl+'/',identifier)
    r = requests.delete(producturl, auth=(username, password))
    try:
        response = r.json()
    except ValueError as e:
        raise saraAdminError("Deleting {} from {} failed with response text: {}".format(identifier, producturl, r.text()))

    if "status" in response:
        if response["status"]=="success":
            if verbose: print("{}: {}".format(producturl, response["message"]))
    elif "ErrorMessage" in response:
        raise saraAdminError("Deleting {} from {} failed: {}".format(identifier, producturl, response["ErrorMessage"]))
    else:    
        # shouldn't end up here?
        raise saraAdminError("Delete {} from {} failed with response: {}.".format(identifer, producturl, response))


class saraAdminError(Exception): pass

